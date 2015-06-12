// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
#include "cuda/Complex.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/fbfft/FBFFTCommon.cuh"
#include "cuda/fbfft/FFT2D32.cuh"

#include <cuda_runtime.h>

#include <cassert>

#include <glog/logging.h>

namespace facebook { namespace cuda { namespace fbfft {

namespace detail {

template <typename T, int Dims>
struct TiledDeviceTensor {
  __host__ __device__
  TiledDeviceTensor() :
      padL(-1), padU(-1), tensor()
    {}

  __host__ __device__
  TiledDeviceTensor(DeviceTensor<T, Dims>& t, int leftPadding, int upPadding) :
      padL(leftPadding),
      padU(upPadding),
      tensor(t.data(), t.sizes(), t.strides())
    {}

  int padL;
  int padU;
  DeviceTensor<T, Dims> tensor;
};

template <typename T, int Dims, int NumTiles>
struct TiledDeviceTensors {
  TiledDeviceTensor<T, Dims> inputs[NumTiles];
  TiledDeviceTensor<T, Dims> weights[NumTiles];
  TiledDeviceTensor<T, Dims> outputs[NumTiles];
};

template<int NumWarps, int NumTilesPerWarp, int FFTSize>
__global__ void updateOutputIteratedKernel(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    size_t numTiles) {

  // constexpr int NumWarps = (NumTiles / NumTilesPerWarp)
  assert(numTiles % NumTilesPerWarp == 0);

  // sharedMem[A] where A must be threadIdx.y == number of FFTs in parallel
  __shared__ float sharedMem[NumWarps][FFTSize][FFTSize + 1];

  const int batch = blockIdx.x;
  const int outputPlane = blockIdx.y;
  const int tileIndexStart = threadIdx.y * NumTilesPerWarp;
  const float invNorm = 1.0f / (FFTSize * FFTSize);

  for (int inputPlane = 0; inputPlane < weight.getSize(1); ++inputPlane) {
    // A1. Read weights (o x i x h x w)
    Complex wei[FFTSize];
    for (int i = 0 ; i < FFTSize; ++i) {
      wei[i] =
        inBounds(i, threadIdx.x, 0, 0, weight) ?
        Complex(weight[outputPlane]
                [inputPlane]
                [i - 0]
                [threadIdx.x - 0].ldg()) :
        Complex(0.0f, 0.0f);
    }

    // B1. In place FFTs via sharedMem (w x h format)
    fbfft2DCore<FFTSize, false>(wei, sharedMem);

    for (int tileIndexOuter = tileIndexStart;
         tileIndexOuter < numTiles; tileIndexOuter += NumTilesPerWarp * NumWarps)
    {{{
          #pragma unroll
          for (int tileIndex = tileIndexOuter;
               tileIndex < tileIndexOuter + NumTilesPerWarp; ++tileIndex)
          {{{
                Complex out[FFTSize];
                // Some extra nesting for readability
                for (int i = 0; i < FFTSize; ++i) {
                  out[i] = 0.0f;
                }

                // A2. Read input (b x i x h x w)
                Complex in[FFTSize];
                const TiledDeviceTensor<float, 4>& tdi = ins[tileIndex];
                for (int i = 0 ; i < FFTSize; ++i) {
                  in[i] =
                    inBounds(i, threadIdx.x, tdi.padU, tdi.padL, tdi.tensor) ?
                    Complex(tdi.tensor[batch]
                            [inputPlane]
                            [i - tdi.padU]
                            [threadIdx.x - tdi.padL].ldg()) :
                    Complex(0.0f, 0.0f);
                }

                // B2. In place FFTs via sharedMem (w x h format)
                fbfft2DCore<FFTSize, false>(in, sharedMem);

                // C. Pointwise multiply and conjugation
                for (int i = 0; i < FFTSize; ++i) {
                  out[i] += in[i] * wei[i].conjugate();
                }

                // D. IFFT out the chunk in place as an fft (i.e. conjugation)
                for (int i = 0; i < FFTSize; ++i) {
                  out[i] = out[i].conjugate();
                }

                fbfft2DCore<FFTSize, true>(out, sharedMem);

                float notIsFirstInputPlane = (inputPlane == 0) ? 0.0f : 1.0f;
                TiledDeviceTensor<float, 4>& tdo = outs[tileIndex];
                for (int i = 0 ; i < FFTSize; ++i) {
                  if (inBounds(i, threadIdx.x, tdo.padU, tdo.padL, tdo.tensor)) {
                    tdo.tensor[batch]
                              [outputPlane]
                              [i - tdo.padU]
                              [threadIdx.x - tdo.padL] =
                      notIsFirstInputPlane *
                      tdo.tensor[batch]
                                [outputPlane]
                                [i - tdo.padU]
                                [threadIdx.x - tdo.padL] +
                      out[i].re() * invNorm;

                  }
                }

              }}} // for tileIndex
        }}} // for tileIndexOuter
  } // for inputPlaneOuter
}


template<int NumWarps, int NumTilesPerWarp, int FFTSize>
__global__ void updateGradInputIteratedKernel(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    size_t numTiles) {

  // sharedMem[A] where A must be threadIdx.y == number of FFTs in parallel
  __shared__ float sharedMem[NumWarps][FFTSize][FFTSize + 1];

  const int batch = blockIdx.x;
  const int inputPlane = blockIdx.y;
  const int tileIndexStart = threadIdx.y * NumTilesPerWarp;
  const float invNorm = 1.0f / (FFTSize * FFTSize);

  for (int outputPlane = 0; outputPlane < weight.getSize(0); ++outputPlane) {
    Complex wei[FFTSize];

    for (int tileIndexOuter = tileIndexStart;
         tileIndexOuter < numTiles; tileIndexOuter += NumTilesPerWarp * NumWarps)
    {{{
          #pragma unroll
          for (int tileIndex = tileIndexOuter;
               tileIndex < tileIndexOuter + NumTilesPerWarp; ++tileIndex)
          {{{
                // Some extra nesting for readability
                Complex in[FFTSize];
                for (int i = 0; i < FFTSize; ++i) {
                  in[i] = 0.0f;
                }

                // A1. Read weights (o x i x h x w)
                for (int i = 0 ; i < FFTSize; ++i) {
                  wei[i] =
                    inBounds(i, threadIdx.x, 0, 0, weight) ?
                    Complex(weight[outputPlane]
                                  [inputPlane]
                                  [i - 0]
                                  [threadIdx.x - 0].ldg()) :
                    Complex(0.0f, 0.0f);
                }

                Complex out[FFTSize];
                // A2. Read input (b x i x h x w)
                const TiledDeviceTensor<float, 4>& tdo = outs[tileIndex];
                for (int i = 0 ; i < FFTSize; ++i) {
                  out[i] =
                    inBounds(i, threadIdx.x, tdo.padU, tdo.padL, tdo.tensor) ?
                    Complex(tdo.tensor[batch]
                                      [outputPlane]
                                      [i - tdo.padU]
                                      [threadIdx.x - tdo.padL].ldg()) :
                    Complex(0.0f, 0.0f);
                }

                // B1. In place FFTs via sharedMem (w x h format)
                fbfft2DCore<FFTSize, false>(wei, sharedMem);

                // B2. In place FFTs via sharedMem (w x h format)
                fbfft2DCore<FFTSize, false>(out, sharedMem);

                // C. Pointwise multiply and conjugation
                for (int i = 0; i < FFTSize; ++i) {
                  in[i] += out[i] * wei[i];
                }

                // D. IFFT out the chunk in place as an fft (i.e. conjugation)
                for (int i = 0; i < FFTSize; ++i) {
                  in[i] = in[i].conjugate();
                }
                fbfft2DCore<FFTSize, true>(in, sharedMem);

                float notIsFirstOutputPlane = (outputPlane == 0) ? 0.0f : 1.0f;
                TiledDeviceTensor<float, 4>& tdi = ins[tileIndex];
                for (int i = 0 ; i < FFTSize; ++i) {
                  if (inBounds(i, threadIdx.x, tdi.padU, tdi.padL, tdi.tensor)) {
                    tdi.tensor[batch]
                              [inputPlane]
                              [i - tdi.padU]
                              [threadIdx.x - tdi.padL] =
                      notIsFirstOutputPlane *
                      tdi.tensor[batch]
                                [inputPlane]
                                [i - tdi.padU]
                                [threadIdx.x - tdi.padL] +
                      in[i].re() * invNorm;

                  }
                }
              }}} // for tileIndex
        }}} // for tileIndexOuter
  } // for outputPlane
}



template<int NumWarps, int NumTilesPerWarp, int FFTSize>
__global__ void accGradParametersIteratedKernel(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    DeviceTensor<float, 4> weight,
    size_t numTiles,
    float scale) {

  // sharedMem[A] where A must be threadIdx.y == number of FFTs in parallel
  __shared__ float sharedMem[NumWarps][FFTSize][FFTSize + 1];

  const int outputPlane = blockIdx.x;
  const int inputPlane = blockIdx.y;
  const int tileIndexStart = threadIdx.y * NumTilesPerWarp;
  const float invNorm = 1.0f / (FFTSize * FFTSize);

  Complex wei[FFTSize];
  Complex in[FFTSize];
  Complex out[FFTSize];

  #pragma unroll
  for (int i = 0; i < FFTSize; ++i) {
    wei[i] = 0.0f;
  }

  for (int batch = 0; batch < ins[0].tensor.getSize(0); ++batch) {
    for (int tileIndexOuter = tileIndexStart;
         tileIndexOuter < numTiles; tileIndexOuter += NumTilesPerWarp * NumWarps)
    {{{
          #pragma unroll
          for (int tileIndex = tileIndexOuter;
               tileIndex < tileIndexOuter + NumTilesPerWarp; ++tileIndex)
          {{{

                // Some extra nesting for readability
                // A1. Read input (b x i x h x w)
                const TiledDeviceTensor<float, 4>& tdi = ins[tileIndex];
                #pragma unroll
                for (int i = 0 ; i < FFTSize; ++i) {
                  in[i] =
                    inBounds(i, threadIdx.x, tdi.padU, tdi.padL, tdi.tensor) ?
                    Complex(tdi.tensor[batch]
                                      [inputPlane]
                                      [i - tdi.padU]
                                      [threadIdx.x - tdi.padL].ldg()) :
                    Complex(0.0f, 0.0f);
                }

                // A2. Read gradOutput (b x i x h x w)
                const TiledDeviceTensor<float, 4>& tdo = outs[tileIndex];
                #pragma unroll
                for (int i = 0 ; i < FFTSize; ++i) {
                  out[i] =
                    inBounds(i, threadIdx.x, tdo.padU, tdo.padL, tdo.tensor) ?
                    Complex(tdo.tensor[batch]
                                      [outputPlane]
                                      [i - tdo.padU]
                                      [threadIdx.x - tdo.padL].ldg()) :
                    Complex(0.0f, 0.0f);
                }


                // B1. In place FFTs via sharedMem (w x h format)
                fbfft2DCore<FFTSize, false>(in, sharedMem);

                // B2. In place FFTs via sharedMem (w x h format)
                fbfft2DCore<FFTSize, false>(out, sharedMem);

                // C. Pointwise multiply and conjugation
                // invNorm and scale only the local contribution
                #pragma unroll
                for (int i = 0; i < FFTSize; ++i) {
                  wei[i] += (in[i] * out[i].conjugate());
                }
              }}} // for tileIndex
        }}} // for tileIndexOuter
  } // for outputPlaneOuter

  // D. IFFT out the chunk in place as an fft (i.e. needs conjugation)
  #pragma unroll
  for (int i = 0; i < FFTSize; ++i) {
    wei[i] = wei[i].conjugate() * (invNorm * scale);
  }
  fbfft2DCore<FFTSize, true>(wei, sharedMem);

  // Need to reduce across threadIdx.y
  #pragma unroll
  for (int i = NumWarps - 1; i >= 0; --i) {
    if (i == threadIdx.y) {
      if (i == NumWarps - 1) {
        #pragma unroll
        for (int j = 0; j < FFTSize; ++j) {
          sharedMem[0][j][threadIdx.x] = wei[j].re();
        }
      } else if (i != 0) {
        #pragma unroll
        for (int j = 0; j < FFTSize; ++j) {
          sharedMem[0][j][threadIdx.x] += wei[j].re();
        }
      } else { // threadIdx.y == 0
        #pragma unroll
        for (int j = 0; j < FFTSize; ++j) {
          wei[j].re() += sharedMem[0][j][threadIdx.x];
        }
      }
    }
    __syncthreads();
  }

  // Only threadIdx.y == 0 writes to memory
  if (threadIdx.y == 0) {
    #pragma unroll
    for (int i = 0 ; i < FFTSize; ++i) {
      if (inBounds(i, threadIdx.x, 0, 0, weight)) {
        weight[outputPlane]
              [inputPlane]
              [i - 0]
              [threadIdx.x - 0] = wei[i].re();
      }
    }
  }
}




#define STATIC_CEIL(N, DIV) (int) ((N + DIV - 1) / DIV)

#define INST_UPDATE_OUTPUT_ITERATED(NUM_WARPS, NUM_TILES_PER_WARP)      \
  if (numTiles % (NUM_WARPS * NUM_TILES_PER_WARP) == 0) {               \
    /*                batch             x     outputPlanes       */     \
    dim3 blocks(batchSize, weight.getSize(0));                          \
    dim3 threads(FFTSize, NUM_WARPS);                                   \
    updateOutputIteratedKernel<NUM_WARPS,                               \
                               NUM_TILES_PER_WARP,                      \
                               FFTSize>                                 \
      <<<blocks, threads, 0, s>>>(ins, outs, weight, numTiles);         \
    return true;                                                        \
  }

#define INST_UPDATE_GRAD_INPUT_ITERATED(NUM_WARPS, NUM_TILES_PER_WARP)  \
  if (numTiles % (NUM_WARPS * NUM_TILES_PER_WARP) == 0) {               \
    /*                batch             x     inputPlanes       */      \
    dim3 blocks(batchSize, weight.getSize(1));                          \
    dim3 threads(FFTSize, NUM_WARPS);                                   \
    updateGradInputIteratedKernel<NUM_WARPS,                            \
                                  NUM_TILES_PER_WARP,                   \
                                  FFTSize>                              \
      <<<blocks, threads, 0, s>>>(ins, outs, weight, numTiles);         \
    return true;                                                        \
  }

#define INST_ACC_GRAD_PARAMETERS_ITERATED(NUM_WARPS, NUM_TILES_PER_WARP) \
  if (numTiles % (NUM_WARPS * NUM_TILES_PER_WARP) == 0) {               \
    /*           outputPlanes    x     inputPlanes          */          \
    dim3 blocks(weight.getSize(0), weight.getSize(1));                  \
    dim3 threads(FFTSize, NUM_WARPS);                                   \
    accGradParametersIteratedKernel<NUM_WARPS,                          \
                                    NUM_TILES_PER_WARP,                 \
                                    FFTSize>                            \
      <<<blocks, threads, 0, s>>>(                                      \
        ins, outs, weight, numTiles, scale);                            \
    return true;                                                        \
  }


// A bit awkward but used directly from lua FFI
typedef struct {
  static const int FFT_UpdateOutput = 0;
  static const int FFT_UpdateGradInput = 1;
  static const int FFT_AccGradParameters = 2;
  int pass;
} FFTConvolutionPassFFI;

template<int FFTSize> bool FFTIteratedConvolution(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    FFTConvolutionPassFFI pass,
    float scale,
    int batchSize,
    size_t numTiles,
    cudaStream_t s);

template<> bool FFTIteratedConvolution<8>(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    FFTConvolutionPassFFI pass,
    float scale,
    int batchSize,
    size_t numTiles,
    cudaStream_t s) {

#define FFTSize 8

  if (pass.pass == pass.FFT_UpdateOutput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);

    // Tune which sizes to use in practice
    INST_UPDATE_OUTPUT_ITERATED(5, 5);
    INST_UPDATE_OUTPUT_ITERATED(4, 4);
    INST_UPDATE_OUTPUT_ITERATED(3, 3);
    INST_UPDATE_OUTPUT_ITERATED(2, 2);
    INST_UPDATE_OUTPUT_ITERATED(1, 1);
  }

  if (pass.pass == pass.FFT_UpdateGradInput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);

    // Tune which sizes to use in practice
    INST_UPDATE_GRAD_INPUT_ITERATED(5, 5);
    INST_UPDATE_GRAD_INPUT_ITERATED(4, 4);
    INST_UPDATE_GRAD_INPUT_ITERATED(3, 3);
    INST_UPDATE_GRAD_INPUT_ITERATED(2, 2);
    INST_UPDATE_GRAD_INPUT_ITERATED(1, 1);
  }

  if (pass.pass == pass.FFT_AccGradParameters) {
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, weight.getSize(1));
    CHECK_LE(1, FFTSize);

    // Tune which sizes to use in practice
    INST_ACC_GRAD_PARAMETERS_ITERATED(1, 1);
  }

#undef FFTSize

  return false;
}

template<> bool FFTIteratedConvolution<16>(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    FFTConvolutionPassFFI pass,
    float scale,
    int batchSize,
    size_t numTiles,
    cudaStream_t s) {

#define FFTSize 16

  if (pass.pass == pass.FFT_UpdateOutput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);

    // Tune which sizes to use in practice
    INST_UPDATE_OUTPUT_ITERATED(5, 5);
    INST_UPDATE_OUTPUT_ITERATED(4, 4);
    INST_UPDATE_OUTPUT_ITERATED(3, 3);
    INST_UPDATE_OUTPUT_ITERATED(2, 2);
    INST_UPDATE_OUTPUT_ITERATED(1, 1);
  }

  if (pass.pass == pass.FFT_UpdateGradInput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);

    // Tune which sizes to use in practice
    INST_UPDATE_GRAD_INPUT_ITERATED(5, 5);
    INST_UPDATE_GRAD_INPUT_ITERATED(4, 4);
    INST_UPDATE_GRAD_INPUT_ITERATED(3, 3);
    INST_UPDATE_GRAD_INPUT_ITERATED(2, 2);
    INST_UPDATE_GRAD_INPUT_ITERATED(1, 1);
  }

  if (pass.pass == pass.FFT_AccGradParameters) {
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, weight.getSize(1));
    CHECK_LE(1, FFTSize);

    // Tune which sizes to use in practice
    INST_ACC_GRAD_PARAMETERS_ITERATED(1, 1);
  }

#undef FFTSize

  return false;
}

template<> bool FFTIteratedConvolution<32>(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    FFTConvolutionPassFFI pass,
    float scale,
    int batchSize,
    size_t numTiles,
    cudaStream_t s) {

#define FFTSize 32

  if (pass.pass == pass.FFT_UpdateOutput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);

    // Tune which sizes to use in practice
    INST_UPDATE_OUTPUT_ITERATED(5, 5);
    INST_UPDATE_OUTPUT_ITERATED(4, 4);
    INST_UPDATE_OUTPUT_ITERATED(3, 3);
    INST_UPDATE_OUTPUT_ITERATED(2, 2);
    INST_UPDATE_OUTPUT_ITERATED(1, 1);
  }

  if (pass.pass == pass.FFT_UpdateGradInput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);

    // Tune which sizes to use in practice
    INST_UPDATE_GRAD_INPUT_ITERATED(5, 5);
    INST_UPDATE_GRAD_INPUT_ITERATED(4, 4);
    INST_UPDATE_GRAD_INPUT_ITERATED(3, 3);
    INST_UPDATE_GRAD_INPUT_ITERATED(2, 2);
    INST_UPDATE_GRAD_INPUT_ITERATED(1, 1);
  }

  if (pass.pass == pass.FFT_AccGradParameters) {
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, weight.getSize(1));
    CHECK_LE(1, FFTSize);

    // Tune which sizes to use in practice
    INST_ACC_GRAD_PARAMETERS_ITERATED(1, 1);
  }

#undef FFTSize

  return false;
}

#undef STATIC_CEIL
#undef INST_UPDATE_OUTPUT_ITERATED
#undef INST_UPDATE_GRAD_INPUT_ITERATED
#undef INST_ACC_GRAD_PARAMETERS_ITERATED

}}}}
