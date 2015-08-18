// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

// Julien Demouth's implementation of the Volkov strategy showed it is
// better to fully unroll by hand and have as much stuff as possible
// compile to immediate instructions.
// 2008. Volkov and Kazian, Fitting FFT onto the G80 Architecture
//
// Write our own using a mix of Volkov for vertical FFTs and
// [1412.7580] Fast Convolutional Nets With fbfft
// for horizontal FFTs.
// This trades off shared memory usage for shuffle instructions in the
// horizontal step.

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
#include "cuda/Complex.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/fbfft/FBFFTCommon.cuh"

#define ENABLE_CUDA_DEBUG
#include "cuda/CudaDebugUtils.cuh"

#include <cuda_runtime.h>

#include <cassert>

namespace facebook { namespace cuda { namespace fbfft {

namespace detail {

template< int > __device__ inline int bitReversal(int);

template<> __device__ inline int bitReversal<4>(int i)
{
  switch (i) {
    case  0:  return 0;
    case  1:  return 2;
    case  2:  return 1;
    case  3:  return 3;
    default:  return 0;
  }
}

template<> __device__ inline int bitReversal<8>(int i)
{
  switch (i) {
    case  0:  return 0;
    case  1:  return 4;
    case  2:  return 2;
    case  3:  return 6;
    case  4:  return 1;
    case  5:  return 5;
    case  6:  return 3;
    case  7:  return 7;
    default:  return 0;
  }
}

template<> __device__ inline int bitReversal<16>(int i)
{
  switch (i) {
    case  0:  return 0;
    case  1:  return 8;
    case  2:  return 4;
    case  3:  return 12;
    case  4:  return 2;
    case  5:  return 10;
    case  6:  return 6;
    case  7:  return 14;
    case  8:  return 1;
    case  9:  return 9;
    case 10:  return 5;
    case 11:  return 13;
    case 12:  return 3;
    case 13:  return 11;
    case 14:  return 7;
    case 15:  return 15;
    default:  return 0;
  }
}

template<> __device__ inline int bitReversal<32>(int i)
{
  switch (i) {
    case  0:  return 0;
    case  1:  return 16;
    case  2:  return 8;
    case  3:  return 24;
    case  4:  return 4;
    case  5:  return 20;
    case  6:  return 12;
    case  7:  return 28;
    case  8:  return 2;
    case  9:  return 18;
    case  10:  return 10;
    case  11:  return 26;
    case  12:  return 6;
    case  13:  return 22;
    case  14:  return 14;
    case  15:  return 30;
    case  16:  return 1;
    case  17:  return 17;
    case  18:  return 9;
    case  19:  return 25;
    case  20:  return 5;
    case  21:  return 21;
    case  22:  return 13;
    case  23:  return 29;
    case  24:  return 3;
    case  25:  return 19;
    case  26:  return 11;
    case  27:  return 27;
    case  28:  return 7;
    case  29:  return 23;
    case  30:  return 15;
    case  31:  return 31;
    default:  return 0;
  }
}

__device__ inline void FFT2(Complex &a, Complex &b)
{
  float t;
  t = a.re(); a.re() += b.re(); b.re() = t - b.re();
  t = a.im(); a.im() += b.im(); b.im() = t - b.im();
}

__device__ inline void swap(Complex& a, Complex& b) {
  Complex t = a;
  a = b;
  b = t;
}

template<int FFTSize>
__device__ __forceinline__ void swapHorizontal(Complex& a) {
  int LogFFTSize = cuda::getMSB<FFTSize>();
  a = shfl(a, reverse(threadIdx.x, LogFFTSize), FFTSize);
}

__device__ inline void FFT4(
    Complex &a0, Complex &a1, Complex &a2, Complex &a3) {
  FFT2(a0, a2);
  FFT2(a1, a3);

  a3 = a3 * FBFFT32_CEXPF_G.conjugate(); // e(2pi / 32 . (-2 . 8))

  FFT2(a0, a1);
  FFT2(a2, a3);
}

__device__ inline void FFT8(Complex* a)
{
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    FFT2(a[i], a[4 + i]);
  }

  a[5] *= FBFFT32_CEXPF_8.conjugate(); // e(-2pi / 8 .1)
  a[6] *= FBFFT32_CEXPF_G.conjugate(); // e(-2pi / 8 . 2)
  a[7] *= FBFFT32_CEXPF_O.conjugate(); // e(-2pi / 8 . 3)

  FFT4(a[ 0], a[ 1], a[ 2], a[ 3]);
  FFT4(a[ 4], a[ 5], a[ 6], a[ 7]);
}

__device__ inline void FFT16(Complex* a)
{
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    FFT2(a[i], a[8 + i]);
  }

  a[9]  *= FBFFT32_CEXPF_4.conjugate(); // e(-2pi / 16 . 1)
  a[10] *= FBFFT32_CEXPF_8.conjugate(); // e(-2pi / 16 . 2)
  a[11] *= FBFFT32_CEXPF_C.conjugate(); // e(-2pi / 16 . 3)
  a[12] *= FBFFT32_CEXPF_G.conjugate(); // e(-2pi / 16 . 4)
  a[13] *= FBFFT32_CEXPF_K.conjugate(); // e(-2pi / 16 . 5)
  a[14] *= FBFFT32_CEXPF_O.conjugate(); // e(-2pi / 16 . 6)
  a[15] *= FBFFT32_CEXPF_S.conjugate(); // e(-2pi / 16 . 7)

  FFT8(a);
  FFT8(a + 8);
}


__device__ inline void FFT32(Complex* a)
{
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    FFT2(a[i], a[16 + i]);
  }

  a[17] *= FBFFT32_CEXPF_2.conjugate(); // e(-2pi / 32 . 1)
  a[18] *= FBFFT32_CEXPF_4.conjugate(); // e(-2pi / 32 . 2)
  a[19] *= FBFFT32_CEXPF_6.conjugate(); // e(-2pi / 32 . 3)
  a[20] *= FBFFT32_CEXPF_8.conjugate(); // e(-2pi / 32 . 4)
  a[21] *= FBFFT32_CEXPF_A.conjugate(); // e(-2pi / 32 . 5)
  a[22] *= FBFFT32_CEXPF_C.conjugate(); // e(-2pi / 32 . 6)
  a[23] *= FBFFT32_CEXPF_E.conjugate(); // e(-2pi / 32 . 7)
  a[24] *= FBFFT32_CEXPF_G.conjugate(); // e(-2pi / 32 . 8)
  a[25] *= FBFFT32_CEXPF_I.conjugate(); // e(-2pi / 32 . 9)
  a[26] *= FBFFT32_CEXPF_K.conjugate(); // e(-2pi / 32 . 10)
  a[27] *= FBFFT32_CEXPF_M.conjugate(); // e(-2pi / 32 . 11)
  a[28] *= FBFFT32_CEXPF_O.conjugate(); // e(-2pi / 32 . 12)
  a[29] *= FBFFT32_CEXPF_Q.conjugate(); // e(-2pi / 32 . 13)
  a[30] *= FBFFT32_CEXPF_S.conjugate(); // e(-2pi / 32 . 14)
  a[31] *= FBFFT32_CEXPF_U.conjugate(); // e(-2pi / 32 . 15)

  FFT16(a);
  FFT16(a + 16);
}

template<int N>
__device__ inline void fft2dVertical(Complex *a);

template<> void fft2dVertical<4>(Complex* a) {
  FFT4(a[0], a[1], a[2], a[3]);
}

template<> void fft2dVertical<8>(Complex* a) {
  FFT8(a);
}

template<> void fft2dVertical<16>(Complex* a) {
  FFT16(a);
}

template<> void fft2dVertical<32>(Complex* a) {
  FFT32(a);
}

//////////////////////////// FBFFT Generic ////////////////////////////////
template <int FFTSize, int BatchesPerBlock, bool Hermitian = false>
__device__ inline void fbfft2DVerticalCoreForward(Complex* a) {
  // Vertical FFT: real FFTs as complex
  // Vertical FFT: bit reversal
  // Let the compiler unroll and optimize
  if (Hermitian) {
    fft2dVertical<FFTSize / 2>(a);
#pragma unroll
    for (int i = 0; i < FFTSize / 2; ++i) {
      if (i < detail::bitReversal<FFTSize / 2>(i)) {
        // Avoid double swap
        swap(a[i], a[detail::bitReversal<FFTSize / 2>(i)]);
      }
    }

    if (Hermitian) {
      Complex r[FFTSize / 2 + 1];
      r[0] = Complex(a[0].re() + a[0].im(), 0.0f);
#pragma unroll
      for (int i = 1; i < FFTSize / 2; ++i) {
        float xpr, xmr, xpi, xmi;
        xpr = 0.5f * (a[i].re() + a[FFTSize / 2 - i].re());
        xmr = 0.5f * (a[i].re() - a[FFTSize / 2 - i].re());
        xpi = 0.5f * (a[i].im() + a[FFTSize / 2 - i].im());
        xmi = 0.5f * (a[i].im() - a[FFTSize / 2 - i].im());
        // cos, sin (i * pi / (FFTSize / 2))
        float cosf = cos<FFTSize / 2>(i);
        float sinf = sin<FFTSize / 2>(i);
        r[i] = Complex(xpr + cosf * xpi - sinf * xmr,
                       xmi - sinf * xpi - cosf * xmr);
      }
      r[FFTSize / 2] = Complex(a[0].re() - a[0].im(), 0.0f);
#pragma unroll
      for (int i = 0 ; i < FFTSize / 2 + 1; ++i) {
        a[i] = r[i];
      }
    }
  } else {
    fft2dVertical<FFTSize>(a);
#pragma unroll
    for (int i = 0; i < FFTSize; ++i) {
      if (i < detail::bitReversal<FFTSize>(i)) {
        // Avoid double swap
        swap(a[i], a[detail::bitReversal<FFTSize>(i)]);
      }
    }
  }

  // Prepare horizontal FFT
  // Twiddles is the same as for 1D but fully data parallel across threadIdx.y
  FFT1DRoots<FFTSize> roots;
  roots.template twiddles<true>();

  constexpr int UB = (Hermitian) ? FFTSize / 2 + 1 : FFTSize / 2 + 1;

  // Horizontal FFT: complex FFTs as complex
#pragma unroll
  for (int i = 0; i < UB; ++i) {
    decimateInFrequency1DWarp<FFTSize>(a[i], roots[0]);
  }

  // Horizontal FFT: bit reversal across threads
#pragma unroll
  for (int i = 0; i < UB; ++i) {
    swapHorizontal<FFTSize>(a[i]);
  }

}


template <int FFTSize, int BatchesPerBlock, bool Hermitian = false>
__device__ inline void fbfft2DVerticalCoreInverse(Complex* a) {
  // Prepare horizontal FFT
  // Twiddles is the same as for 1D but fully data parallel across threadIdx.y
  FFT1DRoots<FFTSize> roots;
  roots.template twiddles<true>();

  // Horizontal FFT: complex FFTs as complex
#pragma unroll
  for (int i = 0; i < FFTSize / 2 + 1; ++i) {
    decimateInFrequency1DWarp<FFTSize>(a[i], roots[0]);
  }

  // Horizontal FFT: bit reversal across threads
#pragma unroll
  for (int i = 0; i < FFTSize / 2 + 1; ++i) {
    swapHorizontal<FFTSize>(a[i]);
  }

  #pragma unroll
  for (int i = FFTSize / 2 + 1; i < FFTSize; ++i) {
    a[i] = a[FFTSize - i].conjugate();
  }

  // Vertical FFT: real FFTs as complex
  fft2dVertical<FFTSize>(a);

  // Vertical FFT: bit reversal
  // Let the compiler unroll and optimize
#pragma unroll
  for (int i = 0; i < FFTSize; ++i) {
    if (i < detail::bitReversal<FFTSize>(i)) {
      // Avoid double swap
      swap(a[i], a[detail::bitReversal<FFTSize>(i)]);
    }
  }
}

// One single implementation is enough for all cases.
// This implementation does not use the full Hermitian symmetry to reduce
// flops (i.e. does not do 2 real FFTs as 1 complex or 1 2N real FFT as 1 N
// complex FFT).
// It is however not wasteful: only FFTSize / 2 + 1 outputs are needed and
// computed along the y dimension.
//
// After further investigation, the name of the game is reduction of
// registers. Savings flops by  Hermitian symmetry is essentially useless
// since the GPU is completely memory bound.
// What counts is the reduction of number of registers needed.
// For this particular purpose, 1 2N real FFT as 1 N complex FFT is a better
// scheme than 2 real FFTs as 1 complex.
//
// This is not implemented here.
// Without any Hermitian symmetry, we achieve between 185 and 210 GB / s.
template <int BatchDims, int FFTSize, int BatchesPerBlock>
__device__ __forceinline__ void fbfft2DVertical(
    DeviceTensor<float, BatchDims + 2> real,
    DeviceTensor<float, BatchDims + 3> complexAsFloat,
    const int padL,
    const int padU) {
  static_assert(FFTSize % 8 == 0,
                "FBFFT supported only for sizes 8, 16, 32 atm");
  static_assert(BatchesPerBlock >= 1, "BatchesPerBlock should be >= 1");
  assert(gridDim.z == 1);
  assert(blockDim.z == 1);

  const int batch =
    BatchesPerBlock * (blockIdx.x * gridDim.y + blockIdx.y ) + threadIdx.y;

  // Early exit if we would overflow
  if (batch >= real.getSize(0)) {
    return;
  }

  constexpr bool Hermitian = true;
  if (Hermitian) {
    // Perform 2 FFTs in place
    Complex a[FFTSize / 2  + 1];
    // A. read data in
    // TODO: read as float2
    // TODO: f16 implementation
#pragma unroll
    for (int i = 0 ; i < FFTSize; i += 2) {
      float f1 = inBounds(i, threadIdx.x, padU, padL, real) ?
        real[batch][i - padU][threadIdx.x - padL].ldg() : 0.0f;
      float f2 = inBounds(i + 1, threadIdx.x, padU, padL, real) ?
        real[batch][i + 1 - padU][threadIdx.x - padL].ldg() : 0.0f;
      a[i / 2] = Complex(f1, f2);
    }

    // B. - F.
    fbfft2DVerticalCoreForward<FFTSize, BatchesPerBlock, true>(a);

#pragma unroll
    for (int i = 0 ; i < FFTSize / 2 + 1; ++i) {
      complexAsFloat[batch][i][threadIdx.x].template as<Complex>() = a[i];
    }
  }

}

template <int BatchDims, int FFTSize, int BatchesPerBlock>
__device__ __forceinline__ void fbifft2DVertical(
    DeviceTensor<Complex, BatchDims + 2> complexSrc,
    DeviceTensor<float, BatchDims + 2> realDst,
    const int padL,
    const int padU) {

  const int batch =
    BatchesPerBlock * (blockIdx.x * gridDim.y + blockIdx.y ) + threadIdx.y;

  // Early exit if we would overflow
  if (batch >= realDst.getSize(0)) {
    return;
  }

  // Perform 2 FFTs in place
  Complex a[FFTSize];
  // A. read data in
  // TODO: read as float2
  // TODO: f16 implementation
  #pragma unroll
  for (int i = 0 ; i < FFTSize / 2 + 1; ++i) {
    a[i] = complexSrc[batch][i][threadIdx.x].data()->conjugate();
  }

  // B. - F.
  fbfft2DVerticalCoreInverse<FFTSize, BatchesPerBlock, false>(a);

  // C. Write the results back to memory.
  // No need for conjugation as we know we have real results.
  for (int i = 0 ; i < FFTSize; ++i) {
    if (inBounds(i, threadIdx.x, padU, padL, realDst)) {
      realDst[batch][i - padU][threadIdx.x - padL] = a[i].re();
    }
  }
}


template <int BatchDims, int BatchesPerBlock>
__global__ void fbfft2DVertical_8(
    DeviceTensor<float, BatchDims + 2> real,
    DeviceTensor<float, BatchDims + 3> complexAsFloat,
    const int padL,
    const int padU) {
  fbfft2DVertical<BatchDims, 8, BatchesPerBlock>(
    real, complexAsFloat, padL, padU);
}

template <int BatchDims, int BatchesPerBlock>
__global__ void fbfft2DVertical_16(
    DeviceTensor<float, BatchDims + 2> real,
    DeviceTensor<float, BatchDims + 3> complexAsFloat,
    const int padL,
    const int padU) {
  fbfft2DVertical<BatchDims, 16, BatchesPerBlock>(
    real, complexAsFloat, padL, padU);
}

template <int BatchDims, int BatchesPerBlock>
__global__ void fbfft2DVertical_32(
    DeviceTensor<float, BatchDims + 2> real,
    DeviceTensor<float, BatchDims + 3> complexAsFloat,
    const int padL,
    const int padU) {
  fbfft2DVertical<BatchDims, 32, BatchesPerBlock>(
    real, complexAsFloat, padL, padU);
}

template <int BatchDims, int BatchesPerBlock>
__global__ void fbifft2DVertical_8(
    DeviceTensor<Complex, BatchDims + 2> complexSrc,
    DeviceTensor<float, BatchDims + 2> realDst,
    const int padL,
    const int padU) {
  fbifft2DVertical<BatchDims, 8, BatchesPerBlock>(
    complexSrc, realDst, padL, padU);
}

template <int BatchDims, int BatchesPerBlock>
__global__ void fbifft2DVertical_16(
    DeviceTensor<Complex, BatchDims + 2> complexSrc,
    DeviceTensor<float, BatchDims + 2> realDst,
    const int padL,
    const int padU) {
  fbifft2DVertical<BatchDims, 16, BatchesPerBlock>(
    complexSrc, realDst, padL, padU);
}

template <int BatchDims, int BatchesPerBlock>
__global__ void fbifft2DVertical_32(
    DeviceTensor<Complex, BatchDims + 2> complexSrc,
    DeviceTensor<float, BatchDims + 2> realDst,
    const int padL,
    const int padU) {
  fbifft2DVertical<BatchDims, 32, BatchesPerBlock>(
    complexSrc, realDst, padL, padU);
}

}}}}
