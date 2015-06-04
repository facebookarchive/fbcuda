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
  TiledDeviceTensor() :
      padL(-1), padU(-1), tensor()
    {}

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

template<int NumTiles, int FFTSize>
__global__ void updateOutputIteratedKernel(
    TiledDeviceTensors<float, 4, NumTiles> t,
    const DeviceTensor<float, 4> weight) {

  // sharedMem[A] where A must be threadIdx.y == number of FFTs in parallel
  __shared__ float sharedMem[1][FFTSize][FFTSize + 1];

  const int batch = blockIdx.x;
  const int outputPlane = blockIdx.y;
  const float invNorm = 1.0f / (FFTSize * FFTSize);

  for (int inputPlane = 0; inputPlane < weight.getSize(1); ++inputPlane) {
    Complex wei[FFTSize];
    for (int tileIndex = 0; tileIndex < NumTiles; ++tileIndex)
    {{{
          // Some extra nesting for readability
          Complex out[FFTSize];
          for (int i = 0; i < FFTSize; ++i) {
            out[i] = 0.0f;
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

          Complex in[FFTSize];
          // A2. Read input (b x i x h x w)
          const TiledDeviceTensor<float, 4>& tdi = t.inputs[tileIndex];
          for (int i = 0 ; i < FFTSize; ++i) {
            in[i] =
              inBounds(i, threadIdx.x, tdi.padU, tdi.padL, tdi.tensor) ?
              Complex(tdi.tensor[batch]
                      [inputPlane]
                      [i - tdi.padU]
                      [threadIdx.x - tdi.padL].ldg()) :
              Complex(0.0f, 0.0f);
          }

          // B1. In place FFTs via sharedMem (w x h format)
          fbfft2DCore<FFTSize, false>(wei, sharedMem);

          // B2. In place FFTs via sharedMem (w x h format)
          fbfft2DCore<FFTSize, false>(in, sharedMem);

          // C. Pointwise multiply and conjugation
          for (int i = 0; i < FFTSize; ++i) {
            out[i] += in[i] * wei[i].conjugate();
          }

          // D. IFFT out the chunk in place as an fft (i.e. needs conjugation)
          for (int i = 0; i < FFTSize; ++i) {
            out[i] = out[i].conjugate();
          }
          fbfft2DCore<FFTSize, true>(out, sharedMem);

          float notIsFirstInputPlane = (inputPlane == 0) ? 0.0f : 1.0f;
          TiledDeviceTensor<float, 4>& tdo = t.outputs[tileIndex];
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
                          [threadIdx.x - tdo.padL] + out[i].re() * invNorm;

            }
          }

    }}} // for tileIndex
  } // for inputPlaneOuter
}


template<int NumTiles, int FFTSize>
__global__ void updateGradInputIteratedKernel(
    TiledDeviceTensors<float, 4, NumTiles> t,
    const DeviceTensor<float, 4> weight) {

  // sharedMem[A] where A must be threadIdx.y == number of FFTs in parallel
  __shared__ float sharedMem[1][FFTSize][FFTSize + 1];

  const int batch = blockIdx.x;
  const int inputPlane = blockIdx.y;
  const float invNorm = 1.0f / (FFTSize * FFTSize);

  for (int outputPlane = 0; outputPlane < weight.getSize(0); ++outputPlane) {
    Complex wei[FFTSize];
    for (int tileIndex = 0; tileIndex < NumTiles; ++tileIndex)
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
          const TiledDeviceTensor<float, 4>& tdo = t.outputs[tileIndex];
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

          // D. IFFT out the chunk in place as an fft (i.e. needs conjugation)
          for (int i = 0; i < FFTSize; ++i) {
            in[i] = in[i].conjugate();
          }
          fbfft2DCore<FFTSize, true>(in, sharedMem);

          float notIsFirstOutputPlane = (outputPlane == 0) ? 0.0f : 1.0f;
          TiledDeviceTensor<float, 4>& tdi = t.inputs[tileIndex];
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
  } // for outputPlane
}



template<int NumTiles, int NumTilesPerWarp, int FFTSize>
__global__ void accGradParametersIteratedKernel(
    TiledDeviceTensors<float, 4, NumTiles> t,
    DeviceTensor<float, 4> weight,
    float scale,
    bool firstIter) {

  assert(FFTSize == 16);
  // constexpr int NumWarps = (NumTiles / NumTilesPerWarp)
  assert(NumTiles % NumTilesPerWarp == 0);

  // sharedMem[A] where A must be threadIdx.y == number of FFTs in parallel
  __shared__ float sharedMem[(NumTiles / NumTilesPerWarp)][FFTSize][FFTSize + 1];

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

  for (int batch = 0; batch < t.inputs[0].tensor.getSize(0); ++batch) {
    #pragma unroll
    for (int tileIndex = tileIndexStart;
         tileIndex < tileIndexStart + NumTilesPerWarp; ++tileIndex)
    {{{   // Some extra nesting for readability
          // A1. Read input (b x i x h x w)
          const TiledDeviceTensor<float, 4>& tdi = t.inputs[tileIndex];
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
          const TiledDeviceTensor<float, 4>& tdo = t.outputs[tileIndex];
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
  } // for outputPlaneOuter

  // D. IFFT out the chunk in place as an fft (i.e. needs conjugation)
  #pragma unroll
  for (int i = 0; i < FFTSize; ++i) {
    wei[i] = wei[i].conjugate() * (invNorm * scale);
  }
  fbfft2DCore<FFTSize, true>(wei, sharedMem);

  // Need to reduce across threadIdx.y
  #pragma unroll
  for (int i = (NumTiles / NumTilesPerWarp) - 1; i >= 0; --i) {
    if (i == threadIdx.y) {
      if (i == (NumTiles / NumTilesPerWarp) - 1) {
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
    if (firstIter) {
      #pragma unroll
      for (int i = 0 ; i < FFTSize; ++i) {
        if (inBounds(i, threadIdx.x, 0, 0, weight)) {
          weight[outputPlane]
                [inputPlane]
                [i - 0]
                [threadIdx.x - 0] = wei[i].re();
        }
      }
    } else {
      #pragma unroll
      for (int i = 0 ; i < FFTSize; ++i) {
        if (inBounds(i, threadIdx.x, 0, 0, weight)) {
          weight[outputPlane]
                [inputPlane]
                [i - 0]
                [threadIdx.x - 0] += wei[i].re();
        }
      }
    }
  }

}


// A bit awkward but used directly from lua FFI
typedef struct {
  static const int FFT_UpdateOutput = 0;
  static const int FFT_UpdateGradInput = 1;
  static const int FFT_AccGradParameters = 2;
  int pass;
} FFTConvolutionPassFFI;

template<int NumTiles, int FFTSize>
bool FFTIteratedConvolution(
    TiledDeviceTensors<float, 4, NumTiles> t,
    const DeviceTensor<float, 4> weight,
    FFTConvolutionPassFFI pass,
    float scale,
    bool firstIter,
    cudaStream_t s) {

  if (pass.pass == pass.FFT_UpdateOutput) {
    CHECK_LE(1, t.inputs[0].tensor.getSize(0));
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);
    //                  batch              x     outputPlanes
    dim3 blocks(t.inputs[0].tensor.getSize(0), weight.getSize(0));
    dim3 threads(FFTSize, 1);
    updateOutputIteratedKernel<NumTiles, FFTSize>
      <<<blocks, threads, 0, s>>>(t, weight);
    return true;
  }

  if (pass.pass == pass.FFT_UpdateGradInput) {
    CHECK_LE(1, t.inputs[0].tensor.getSize(0));
    CHECK_LE(1, weight.getSize(1));
    CHECK_LE(1, FFTSize);
    //                  batch              x     inputPlanes
    dim3 blocks(t.inputs[0].tensor.getSize(0), weight.getSize(1));
    dim3 threads(FFTSize, 1);
    updateGradInputIteratedKernel<NumTiles, FFTSize>
      <<<blocks, threads, 0, s>>>(t, weight);
    return true;
  }

  if (pass.pass == pass.FFT_AccGradParameters) {
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, weight.getSize(1));
    CHECK_LE(1, FFTSize);

#define STATIC_CEIL(N, DIV) (int) ((N + DIV - 1) / DIV)

#define INSTANTIATE_ACC_GRAD_PARAMETERS_ITERATED(NUM_WARPS)             \
    if (NumTiles % NUM_WARPS == 0) {                                    \
      /*           outputPlanes    x     inputPlanes          */        \
      dim3 blocks(weight.getSize(0), weight.getSize(1));                \
      dim3 threads(FFTSize, NUM_WARPS);                                 \
      accGradParametersIteratedKernel<NumTiles,                         \
                                      /*static guarantee >= 1*/         \
                                      STATIC_CEIL(NumTiles, NUM_WARPS), \
                                      FFTSize>                          \
        <<<blocks, threads, 0, s>>>(t, weight, scale, firstIter);       \
      return true;                                                      \
    }                                                                   \

    // Tune which sizes to use in practice
    // INSTANTIATE_ACC_GRAD_PARAMETERS_ITERATED(4); Bug here solved in the
    // subsequent diff
    INSTANTIATE_ACC_GRAD_PARAMETERS_ITERATED(1);
  }

  return false;
}

}}}}
