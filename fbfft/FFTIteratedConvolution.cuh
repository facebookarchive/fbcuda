// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// Notes:
//
// The following implementations are prototypes, do not exploit any reuse and
// currently have bad performance. Their purpose is to be a baseline for
// exploring the feasibility of single fused kernels that do not require any
// device memory.
//
// The general structure of the code is a 5-D loop which exhibits reuse
// across:
//   - tiles, batches for weights
//   - outputplanes for inputs
//   - inputplanes for outputs
////
// When implemented with memory reuse in mind, these kernels stress out the
// GPU resources.
// For UpdateOutput, the 2 stressing factors are:
//   - amount of shared memory required for a single block which is given by:
//     InputPlUnroll x OutputPlUnroll x FFTSize / 2 x FFTSize x sizeof(Complex)
//     In practice, for FFT sizes 32x32, a 4 x 4 unroll of InputPlUnroll x
//     OutputPlUnroll requires 65K and does not fit in shared memory.
//     So there is a tradeoff between amount of reuse, what fits into shared
//     memory and precomputing weights in global memory.
//   - amount of register used, number of threads in a threadblock within the
//     limit of the 65K registers.
//     A block has FFTSize x OutputPlUnroll x BatchUnroll threads.
//     By limiting the kernel to use only 128 registers for FFT 32, we can fit
//     32 x 2 x 8 threads in a block within the 64K register K40m budget.
//
// 32x32:
//   This kernel takes up a lot of registers, by squeezing them we get to
//   128 per thread. So we can only hope for 1 block per SM at any point.
//   We must enforce the constrtains:
//   FFTSize, BatchUnroll, InputUnroll, OutputUnroll
//   FFTSize x BatchUnroll x OutputUnroll x 128 <= 65K
//   FFTSize x FFTSize / 2 x InputUnroll x OutputUnroll x 8 <= 40K
//
// 16x16:
//   This kernel can be squeezed within 64 registers per thread.
//   So we can only aim for 2 blocks per SM.
//   We must enforce the constrtains:
//   FFTSize, BatchUnroll, InputUnroll, OutputUnroll
//   FFTSize x BatchUnroll x OutputUnroll x 64 <= 32K
//   FFTSize x FFTSize / 2 x InputUnroll x OutputUnroll x 8 <= 20K
//
// 8x8:
//   This kernel spills badly within 32 registers per thread...
//   But it does fit within 40 registers.
//
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
#include "cuda/Complex.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/fbfft/FBFFTCommon.cuh"
#include "cuda/fbfft/FFT2D32.cuh"

#include <cuda_runtime.h>

#include <cassert>

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

template<int FFTSize>
__device__ void updateOutputIteratedKernel(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    size_t numTiles) {

  constexpr float invNorm = 1.0f / (FFTSize * FFTSize);
  const auto tileIndexStart = blockIdx.x;
  const auto inputPlanes = weight.getSize(1);
  const auto outputPlanes = weight.getSize(0);

  const auto batch = blockIdx.z * blockDim.z + threadIdx.z;
  const auto outputPl = blockIdx.y * blockDim.y + threadIdx.y;

  const auto tileIndex = tileIndexStart;
  const auto& input = ins[tileIndex];
  auto& output = outs[tileIndex];

  // Early exits
  if (batch >= input.tensor.getSize(0)) { return; }
  if (outputPl >= outputPlanes) { return; }

  Complex out[FFTSize / 2];
#pragma unroll
  for (int i = 0 ; i < FFTSize / 2; ++i) {
    out[i] = Complex(0.0f);
  }

  for (int inputPl = 0; inputPl < inputPlanes; ++inputPl) {
    Complex wei[FFTSize / 2];
#pragma unroll
    for (int i = 0 ; i < FFTSize; i += 2) {
      float f1 = inBounds(i, threadIdx.x, 0, 0, weight) ?
        weight
        [outputPl][inputPl][i - 0][threadIdx.x - 0].ldg() :
        0.0f;
      float f2 = inBounds(i + 1, threadIdx.x, 0, 0, weight) ?
        weight
        [outputPl][inputPl][i + 1 - 0][threadIdx.x - 0].ldg() :
        0.0f;
      wei[i / 2] = Complex(f1, f2);
    }

    fbfft2DVerticalCoreForward<FFTSize>(wei);

    Complex inp[FFTSize / 2];
#pragma unroll
    for (int i = 0 ; i < FFTSize; i += 2) {
      float f1 =
        inBounds(i, threadIdx.x, input.padU, input.padL, input.tensor) ?
        input.tensor
        [batch][inputPl][i - input.padU][threadIdx.x - input.padL].ldg() :
        0.0f;
      float f2 =
        inBounds(i + 1, threadIdx.x, input.padU, input.padL, input.tensor) ?
        input.tensor
        [batch][inputPl][i + 1 - input.padU][threadIdx.x - input.padL].ldg() :
        0.0f;
      inp[i / 2] = Complex(f1, f2);
    }

    fbfft2DVerticalCoreForward<FFTSize>(inp);

    // First element packs 2 real into a complex, don't get fooled
    Complex tmpin, tmpwei;
    if (threadIdx.x > 0) {
      Complex otherinp = shfl(inp[0], FFTSize - threadIdx.x, FFTSize);
      Complex inp0  = Complex(0.5f) * (otherinp.conjugate() + inp[0]);
      Complex inpN2 =
        Complex(0.5f) *
        (otherinp.conjugate() - inp[0]).conjugate().transpose();
      Complex otherwei = shfl(wei[0], FFTSize - threadIdx.x, FFTSize);
      Complex wei0  = Complex(0.5f) * (otherwei.conjugate() + wei[0]);
      Complex weiN2 =
        Complex(0.5f) *
        (otherwei.conjugate() - wei[0]).conjugate().transpose();
      Complex out0  = inp0  * wei0.conjugate();
      Complex outN2 = inpN2 * weiN2.conjugate();
      out[0] += out0 + outN2.conjugate().transpose();
    } else {
      out[0] +=
        Complex(inp[0].re() * wei[0].re(),
                inp[0].im() * wei[0].im());
    }

#pragma unroll
    for (int i = 1; i < FFTSize / 2; ++i) {
      out[i] += inp[i] * wei[i].conjugate();
    }
  } // inputPl

  fbfft2DVerticalCoreInverse<FFTSize, true>(out);

  // Write the results back to memory.
  // No need for conjugation as we know we have real results.
#pragma unroll
  for (int i = 0 ; i < FFTSize; i += 2) {
    if (inBounds(i, threadIdx.x, output.padU, output.padL, output.tensor)) {
      output.tensor
        [batch][outputPl][i - output.padU][threadIdx.x - output.padL] =
        out[i / 2].re() * invNorm;
    }
    if (inBounds(i + 1, threadIdx.x, output.padU, output.padL, output.tensor)) {
      output.tensor
        [batch][outputPl][i + 1 - output.padU][threadIdx.x - output.padL] =
        out[i / 2].im() * invNorm;
    }
  }

}



__launch_bounds__(512, 1) // 128 registers
__global__ void updateOutputIteratedKernel32(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    size_t numTiles)
{
  updateOutputIteratedKernel<32>(
    ins,
    outs,
    weight,
    numTiles);
}

__launch_bounds__(1024, 1) // 64 registers
__global__ void updateOutputIteratedKernel16(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    size_t numTiles)
{
  updateOutputIteratedKernel<16>(
    ins,
    outs,
    weight,
    numTiles);
}

__launch_bounds__(425, 3) // 40 registers
__global__ void updateOutputIteratedKernel8(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    size_t numTiles)
{
  updateOutputIteratedKernel<8>(
    ins,
    outs,
    weight,
    numTiles);
}


template<int FFTSize>
__device__ void updateGradInputIteratedKernel(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    size_t numTiles) {

  constexpr float invNorm = 1.0f / (FFTSize * FFTSize);
  const auto tileIndexStart = blockIdx.x;
  const auto inputPlanes = weight.getSize(1);
  const auto outputPlanes = weight.getSize(0);

  const auto batch = blockIdx.z * blockDim.z + threadIdx.z;
  const auto inputPl = blockIdx.y * blockDim.y + threadIdx.y;

  const auto tileIndex = tileIndexStart;
  auto& input = ins[tileIndex];
  const auto& output = outs[tileIndex];

  // Early exits
  if (batch >= input.tensor.getSize(0)) { return; }
  if (inputPl >= weight.getSize(1)) { return; }

  Complex inp[FFTSize / 2];
#pragma unroll
  for (int i = 0 ; i < FFTSize / 2; ++i) {
    inp[i] = Complex(0.0f);
  }

  for (int outputPl = 0; outputPl < outputPlanes; ++outputPl) {
    Complex wei[FFTSize / 2];
#pragma unroll
    for (int i = 0 ; i < FFTSize; i += 2) {
      float f1 = inBounds(i, threadIdx.x, 0, 0, weight) ?
        weight
        [outputPl][inputPl][i - 0][threadIdx.x - 0].ldg() :
        0.0f;
      float f2 = inBounds(i + 1, threadIdx.x, 0, 0, weight) ?
        weight
        [outputPl][inputPl][i + 1 - 0][threadIdx.x - 0].ldg() :
        0.0f;
      wei[i / 2] = Complex(f1, f2);
    }
    fbfft2DVerticalCoreForward<FFTSize>(wei);

    Complex out[FFTSize / 2];
#pragma unroll
    for (int i = 0 ; i < FFTSize; i += 2) {
      float f1 =
        inBounds(i, threadIdx.x, output.padU, output.padL, output.tensor) ?
        output.tensor
        [batch][outputPl][i - output.padU][threadIdx.x - output.padL].ldg() :
        0.0f;
      float f2 =
        inBounds(i + 1, threadIdx.x, output.padU, output.padL, output.tensor) ?
        output.tensor
        [batch][outputPl][i + 1 - output.padU][threadIdx.x - output.padL].ldg() :
        0.0f;
      out[i / 2] = Complex(f1, f2);
    }
    fbfft2DVerticalCoreForward<FFTSize>(out);

    // First element packs 2 real into a complex, don't get fooled
    Complex tmpin, tmpwei;
    if (threadIdx.x > 0) {
      Complex otherout = shfl(out[0], FFTSize - threadIdx.x, FFTSize);
      Complex out0  = Complex(0.5f) * (otherout.conjugate() + out[0]);
      Complex outN2 =
        Complex(0.5f) *
        (otherout.conjugate() - out[0]).conjugate().transpose();
      Complex otherwei = shfl(wei[0], FFTSize - threadIdx.x, FFTSize);
      Complex wei0  = Complex(0.5f) * (otherwei.conjugate() + wei[0]);
      Complex weiN2 =
        Complex(0.5f) *
        (otherwei.conjugate() - wei[0]).conjugate().transpose();
      Complex in0  = out0  * wei0;
      Complex inN2 = outN2 * weiN2;
      inp[0] += in0 + inN2.conjugate().transpose();
    } else {
      inp[0] +=
        Complex(out[0].re() * wei[0].re(),
                out[0].im() * wei[0].im());
    }

#pragma unroll
    for (int i = 1; i < FFTSize / 2; ++i) {
      inp[i] += out[i] * wei[i];
    }

  } // inputPl

  fbfft2DVerticalCoreInverse<FFTSize, true>(inp);

  // Write the results back to memory.
  // No need for conjugation as we know we have real results.
#pragma unroll
  for (int i = 0 ; i < FFTSize; i += 2) {
    if (inBounds(i, threadIdx.x, input.padU, input.padL, input.tensor)) {
      input.tensor
        [batch][inputPl][i - input.padU][threadIdx.x - input.padL] =
        inp[i / 2].re() * invNorm;
    }
    if (inBounds(i + 1, threadIdx.x, input.padU, input.padL, input.tensor)) {
      input.tensor
        [batch][inputPl][i + 1 - input.padU][threadIdx.x - input.padL] =
        inp[i / 2].im() * invNorm;
    }
  }

}

__launch_bounds__(512, 1) // 128 registers
__global__ void updateGradInputIteratedKernel32(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    size_t numTiles)
{
  updateGradInputIteratedKernel<32>(
    ins,
    outs,
    weight,
    numTiles);
}

__launch_bounds__(512, 1) // 128 registers
__global__ void updateGradInputIteratedKernel16(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    size_t numTiles)
{
  updateGradInputIteratedKernel<16>(
    ins,
    outs,
    weight,
    numTiles);
}

__launch_bounds__(1024, 1) // 64 registers
__global__ void updateGradInputIteratedKernel8(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    const DeviceTensor<float, 4> weight,
    size_t numTiles)
{
  updateGradInputIteratedKernel<8>(
    ins,
    outs,
    weight,
    numTiles);
}



template<int FFTSize>
__device__ void accGradParametersIteratedKernel(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    DeviceTensor<float, 4> weight,
    size_t numTiles,
    float scale) {

  const float invNormScale = scale / (FFTSize * FFTSize);
  const auto inputPlanes = weight.getSize(1);
  const auto outputPlanes = weight.getSize(0);

  const auto inputPl = blockIdx.y * blockDim.y + threadIdx.y;
  const auto outputPl = blockIdx.z * blockDim.z + threadIdx.z;

  Complex wei[FFTSize / 2 + 1];
#pragma unroll
  for (int i = 0 ; i < FFTSize / 2; ++i) {
    wei[i] = Complex(0.0f);
  }
  wei[FFTSize / 2] = Complex(0.0f);

  for (int tileIndex = 0; tileIndex < numTiles; ++tileIndex) {

    const auto& input = ins[tileIndex];
    const auto& output = outs[tileIndex];

    // Early exits
    if (inputPl >= inputPlanes) { return; }
    if (outputPl >= outputPlanes) { return; }

    const auto Batches = input.tensor.getSize(0);

    for (int batch = 0; batch < Batches; ++batch) {

      Complex out[FFTSize / 2 + 1];
#pragma unroll
      for (int i = 0 ; i < FFTSize; i += 2) {
        float f1 =
          inBounds(i, threadIdx.x, output.padU, output.padL, output.tensor) ?
          output.tensor
          [batch][outputPl][i - output.padU][threadIdx.x - output.padL].ldg() :
          0.0f;
        float f2 =
          inBounds(i + 1, threadIdx.x, output.padU, output.padL, output.tensor) ?
          output.tensor
          [batch][outputPl][i + 1 - output.padU][threadIdx.x - output.padL].ldg() :
          0.0f;
        out[i / 2] = Complex(f1, f2);
      }
      out[FFTSize / 2] = Complex(0.0f);
      fbfft2DVerticalCoreForward<FFTSize>(out);


      Complex inp[FFTSize / 2 + 1];
#pragma unroll
      for (int i = 0 ; i < FFTSize; i += 2) {
        float f1 =
          inBounds(i, threadIdx.x, input.padU, input.padL, input.tensor) ?
          input.tensor
          [batch][inputPl][i - input.padU][threadIdx.x - input.padL].ldg() :
          0.0f;
        float f2 =
          inBounds(i + 1, threadIdx.x, input.padU, input.padL, input.tensor) ?
          input.tensor
          [batch][inputPl][i + 1 - input.padU][threadIdx.x - input.padL].ldg() :
          0.0f;
        inp[i / 2] = Complex(f1, f2);
      }

      inp[FFTSize / 2] = Complex(0.0f);
      fbfft2DVerticalCoreForward<FFTSize>(inp);

      // First element packs 2 real into a complex, don't get fooled
      Complex tmpin, tmpwei;
      if (threadIdx.x > 0) {
        // Unpack
        Complex otherinp = shfl(inp[0], FFTSize - threadIdx.x, FFTSize);
        inp[FFTSize / 2] = Complex(0.5f) *
          (otherinp.conjugate() - inp[0]).conjugate().transpose();
        inp[0]  = Complex(0.5f) * (otherinp.conjugate() + inp[0]);
        // Unpack
        Complex otherout = shfl(out[0], FFTSize - threadIdx.x, FFTSize);
        out[FFTSize / 2] = Complex(0.5f) *
          (otherout.conjugate() - out[0]).conjugate().transpose();
        out[0]  = Complex(0.5f) * (otherout.conjugate() + out[0]);
      } else {
        inp[FFTSize / 2] = Complex(inp[0].im());
        inp[0]  = Complex(inp[0].re());
        out[FFTSize / 2] = Complex(out[0].im());
        out[0]  = Complex(out[0].re());
      }

#pragma unroll
      for (int i = 0; i < FFTSize / 2 + 1; ++i) {
        wei[i] += inp[i] * out[i].conjugate();
      }
    } // batches
  } // tileIndex

  fbfft2DVerticalCoreInverse<FFTSize, false>(wei);

  // Write the results back to memory.
  // No need for conjugation as we know we have real results.
#pragma unroll
  for (int i = 0 ; i < FFTSize; i += 2) {
    if (inBounds(i, threadIdx.x, 0, 0, weight)) {
      weight[outputPl][inputPl][i - 0][threadIdx.x - 0] =
        wei[i / 2].re() * invNormScale;
    }
    if (inBounds(i + 1, threadIdx.x, 0, 0, weight)) {
      weight[outputPl][inputPl][i + 1 - 0][threadIdx.x - 0] =
        wei[i / 2].im() * invNormScale;
    }
  }
}


__launch_bounds__(512, 1) // 128 registers
__global__ void accGradParametersIteratedKernel32(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    DeviceTensor<float, 4> weight,
    size_t numTiles,
    float scale) {
  accGradParametersIteratedKernel<32>(
    ins,
    outs,
    weight,
    numTiles,
    scale);
}

__launch_bounds__(512, 1) // 128 registers
__global__ void accGradParametersIteratedKernel16(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    DeviceTensor<float, 4> weight,
    size_t numTiles,
    float scale) {
  accGradParametersIteratedKernel<16>(
    ins,
    outs,
    weight,
    numTiles,
    scale);
}

__launch_bounds__(1024, 1) // 64 registers
__global__ void accGradParametersIteratedKernel8(
    TiledDeviceTensor<float, 4>* ins,
    TiledDeviceTensor<float, 4>* outs,
    DeviceTensor<float, 4> weight,
    size_t numTiles,
    float scale) {
  accGradParametersIteratedKernel<8>(
    ins,
    outs,
    weight,
    numTiles,
    scale);
}



constexpr int ceil(int num, int denom) {
  return (num + denom - 1) / denom;
}

#define INST_UPDATE_OUTPUT_ITERATED(FFTSIZE, BUNROLL, IUNROLL, OUNROLL) \
  {                                                                     \
    dim3 blocks(numTiles,                                               \
                ceil(outputPlanes, OUNROLL),                            \
                ceil(batchSize, BUNROLL));                              \
    dim3 threads(FFTSIZE, OUNROLL, BUNROLL);                            \
    updateOutputIteratedKernel##FFTSIZE                                 \
      <<<blocks, threads, 0, s>>>(ins,                                  \
                                  outs,                                 \
                                  weight,                               \
                                  numTiles);                            \
      return true;                                                      \
  }

#define INST_UPDATE_GRAD_INPUT_ITERATED(FFTSIZE, BUNROLL, IUNROLL, OUNROLL) \
  {                                                                     \
    dim3 blocks(numTiles,                                               \
                ceil(inputPlanes, IUNROLL),                             \
                ceil(batchSize, BUNROLL));                              \
    dim3 threads(FFTSIZE, IUNROLL, BUNROLL);                            \
    updateGradInputIteratedKernel##FFTSIZE                              \
      <<<blocks, threads, 0, s>>>(ins,                                  \
                                  outs,                                 \
                                  weight,                               \
                                  numTiles);                            \
      return true;                                                      \
  }

#define INST_ACC_GRAD_PARAMETERS_ITERATED(FFTSIZE, BUNROLL, IUNROLL, OUNROLL) \
  {                                                                     \
    dim3 blocks(1, /* Accumulate so must always be 1! */                \
                ceil(inputPlanes, IUNROLL),                             \
                ceil(outputPlanes, OUNROLL));                           \
    dim3 threads(FFTSIZE, IUNROLL, OUNROLL);                            \
    accGradParametersIteratedKernel##FFTSIZE                            \
      <<<blocks, threads, 0, s>>>(ins,                                  \
                                  outs,                                 \
                                  weight,                               \
                                  numTiles,                             \
                                  scale);                               \
      return true;                                                      \
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

  const auto inputPlanes = weight.getSize(1);
  const auto outputPlanes = weight.getSize(0);

  // Don't forget to init your twiddles
  facebook::cuda::fbfft::initTwiddles();

  constexpr int FFTSize = 8;
  if (pass.pass == pass.FFT_UpdateOutput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);
    INST_UPDATE_OUTPUT_ITERATED(8, 4, 4, 4);
  }

  if (pass.pass == pass.FFT_UpdateGradInput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);
    INST_UPDATE_GRAD_INPUT_ITERATED(8, 4, 4, 4);
  }

  if (pass.pass == pass.FFT_AccGradParameters) {
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, weight.getSize(1));
    CHECK_LE(1, FFTSize);
    INST_ACC_GRAD_PARAMETERS_ITERATED(8, 1, 4, 4);
  }

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

  const auto inputPlanes = weight.getSize(1);
  const auto outputPlanes = weight.getSize(0);

  // Don't forget to init your twiddles
  facebook::cuda::fbfft::initTwiddles();

  constexpr int FFTSize = 16;
  if (pass.pass == pass.FFT_UpdateOutput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);
    INST_UPDATE_OUTPUT_ITERATED(16, 8, 4, 4);
  }

  if (pass.pass == pass.FFT_UpdateGradInput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);
    INST_UPDATE_GRAD_INPUT_ITERATED(16, 8, 4, 4);
  }

  if (pass.pass == pass.FFT_AccGradParameters) {
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, weight.getSize(1));
    CHECK_LE(1, FFTSize);
    INST_ACC_GRAD_PARAMETERS_ITERATED(16, 1, 2, 2);
  }

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

  const auto inputPlanes = weight.getSize(1);
  const auto outputPlanes = weight.getSize(0);

  // Don't forget to init your twiddles
  facebook::cuda::fbfft::initTwiddles();

  constexpr int FFTSize = 32;
  if (pass.pass == pass.FFT_UpdateOutput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);
    INST_UPDATE_OUTPUT_ITERATED(32, 8, 4, 2);
  }

  if (pass.pass == pass.FFT_UpdateGradInput) {
    CHECK_LE(1, numTiles);
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, FFTSize);
    INST_UPDATE_GRAD_INPUT_ITERATED(32, 8, 2, 4);
  }

  if (pass.pass == pass.FFT_AccGradParameters) {
    CHECK_LE(1, weight.getSize(0));
    CHECK_LE(1, weight.getSize(1));
    CHECK_LE(1, FFTSize);
    INST_ACC_GRAD_PARAMETERS_ITERATED(32, 1, 2, 2);
  }

  return false;
}

#undef INST_UPDATE_OUTPUT_ITERATED
#undef INST_UPDATE_GRAD_INPUT_ITERATED
#undef INST_ACC_GRAD_PARAMETERS_ITERATED

}}}}
