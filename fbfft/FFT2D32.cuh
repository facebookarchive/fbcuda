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

template<typename T>
__device__ inline void swap(T& a, T& b) {
  T t = a;
  a = b;
  b = t;
}

__device__ inline void FFT2(Complex &a, Complex &b)
{
  float t;
  t = a.re(); a.re() += b.re(); b.re() = t - b.re();
  t = a.im(); a.im() += b.im(); b.im() = t - b.im();
}

template<int FFTSize>
__device__ __forceinline__ void swapHorizontal(Complex& a) {
  int LogFFTSize = cuda::getMSB<FFTSize>();
  a = shfl(a, reverse(threadIdx.x, LogFFTSize), FFTSize);
}

template<bool forward>
__device__ inline void FFT4(
    Complex &a0, Complex &a1, Complex &a2, Complex &a3) {
  FFT2(a0, a2);
  FFT2(a1, a3);

  a3 = a3 * ((forward) ? cexp<4>(2).conjugate() : cexp<4>(2));

  FFT2(a0, a1);
  FFT2(a2, a3);
}

template<bool forward>
__device__ inline void FFT8(Complex* a)
{
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    FFT2(a[i], a[4 + i]);
  }

  a[5] *= ((forward) ? cexp<8>(2).conjugate() : cexp<8>(2));
  a[6] *= ((forward) ? cexp<8>(4).conjugate() : cexp<8>(4));
  a[7] *= ((forward) ? cexp<8>(6).conjugate() : cexp<8>(6));

  FFT4<forward>(a[ 0], a[ 1], a[ 2], a[ 3]);
  FFT4<forward>(a[ 4], a[ 5], a[ 6], a[ 7]);
}

template<bool forward>
__device__ inline void FFT16(Complex* a)
{
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    FFT2(a[i], a[8 + i]);
  }

  a[9]  *= ((forward) ? cexp<16>(2).conjugate() : cexp<16>(2));
  a[10] *= ((forward) ? cexp<16>(4).conjugate() : cexp<16>(4));
  a[11] *= ((forward) ? cexp<16>(6).conjugate() : cexp<16>(6));
  a[12] *= ((forward) ? cexp<16>(8).conjugate() : cexp<16>(8));
  a[13] *= ((forward) ? cexp<16>(10).conjugate() : cexp<16>(10));
  a[14] *= ((forward) ? cexp<16>(12).conjugate() : cexp<16>(12));
  a[15] *= ((forward) ? cexp<16>(14).conjugate() : cexp<16>(14));

  FFT8<forward>(a);
  FFT8<forward>(a + 8);
}

template<bool forward>
__device__ inline void FFT32(Complex* a)
{
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    FFT2(a[i], a[16 + i]);
  }

  a[17] *= ((forward) ? cexp<32>(2).conjugate() : cexp<32>(2));
  a[18] *= ((forward) ? cexp<32>(4).conjugate() : cexp<32>(4));
  a[19] *= ((forward) ? cexp<32>(6).conjugate() : cexp<32>(6));
  a[20] *= ((forward) ? cexp<32>(8).conjugate() : cexp<32>(8));
  a[21] *= ((forward) ? cexp<32>(10).conjugate() : cexp<32>(10));
  a[22] *= ((forward) ? cexp<32>(12).conjugate() : cexp<32>(12));
  a[23] *= ((forward) ? cexp<32>(14).conjugate() : cexp<32>(14));
  a[24] *= ((forward) ? cexp<32>(16).conjugate() : cexp<32>(16));
  a[25] *= ((forward) ? cexp<32>(18).conjugate() : cexp<32>(18));
  a[26] *= ((forward) ? cexp<32>(20).conjugate() : cexp<32>(20));
  a[27] *= ((forward) ? cexp<32>(22).conjugate() : cexp<32>(22));
  a[28] *= ((forward) ? cexp<32>(24).conjugate() : cexp<32>(24));
  a[29] *= ((forward) ? cexp<32>(26).conjugate() : cexp<32>(26));
  a[30] *= ((forward) ? cexp<32>(28).conjugate() : cexp<32>(28));
  a[31] *= ((forward) ? cexp<32>(30).conjugate() : cexp<32>(30));

  FFT16<forward>(a);
  FFT16<forward>(a + 16);
}

template<int N, bool forward = true>
__device__ inline void fft2dVertical(Complex *a);

template<> void fft2dVertical<4, true>(Complex* a) {
  FFT4<true>(a[0], a[1], a[2], a[3]);
}

template<> void fft2dVertical<4, false>(Complex* a) {
  FFT4<false>(a[0], a[1], a[2], a[3]);
}

template<> void fft2dVertical<8, true>(Complex* a) {
  FFT8<true>(a);
}

template<> void fft2dVertical<8, false>(Complex* a) {
  FFT8<false>(a);
}

template<> void fft2dVertical<16, true>(Complex* a) {
  FFT16<true>(a);
}

template<> void fft2dVertical<16, false>(Complex* a) {
  FFT16<false>(a);
}

template<> void fft2dVertical<32, true>(Complex* a) {
  FFT32<true>(a);
}

template<> void fft2dVertical<32, false>(Complex* a) {
  FFT32<false>(a);
}

//////////////////////////// FBFFT Generic ////////////////////////////////
template <int FFTSize>
__device__ inline void fbfft2DVerticalCoreForward(Complex* a) {
  constexpr int HalfFFTSize = FFTSize / 2;

  // Vertical FFT: real FFTs as complex
  // Vertical FFT: bit reversal
  // Let the compiler unroll and optimize
  fft2dVertical<HalfFFTSize, true>(a);

#pragma unroll
  for (int i = 0; i < HalfFFTSize; ++i) {
    if (i < detail::bitReversal<HalfFFTSize>(i)) {
      // Avoid double swap
      swap(a[i], a[detail::bitReversal<HalfFFTSize>(i)]);
    }
  }

  a[0] = a[0].conjugate() + a[0].transpose();
#pragma unroll
  for (int i = 1; i < HalfFFTSize / 2; ++i) {
    Complex Si = Complex(0.5f) * (a[i] + a[HalfFFTSize - i].conjugate());
    Complex Di = Complex(0.5f) * (a[i] - a[HalfFFTSize - i].conjugate());
    a[i] = Si - Di * cexp<HalfFFTSize>(i).transpose();
    a[HalfFFTSize - i] = Si.conjugate() -
        (-Di.conjugate()) * cexp<HalfFFTSize>(HalfFFTSize - i).transpose();
  }
  a[HalfFFTSize / 2] = a[HalfFFTSize / 2].conjugate();

#if 0

  // Prepare horizontal FFT
  // Twiddles is the same as for 1D but fully data parallel across threadIdx.y
  FFT1DRegisterTwiddles<FFTSize> roots(true);

  // Horizontal FFT: complex FFTs as complex
#pragma unroll
  for (int i = 0; i < HalfFFTSize; ++i) {
    decimateInFrequency1DWarp<FFTSize>(a[i], roots);
  }

#else

  FFT1DRoots<FFTSize> roots;
  roots.template twiddles<true>();
  // Horizontal FFT: complex FFTs as complex
#pragma unroll
  for (int i = 0; i < HalfFFTSize; ++i) {
    decimateInFrequency1DWarp<FFTSize>(a[i], roots[0]);
  }

#endif

  // With a proper DIT / DIF this could disappear and save us some shuffles
  // Horizontal FFT: bit reversal across threads
#pragma unroll
  for (int i = 0; i < HalfFFTSize; ++i) {
    swapHorizontal<FFTSize>(a[i]);
  }

}

// Only unpack IFFT supported atm
template <int FFTSize, bool PackedInput = false>
__device__ inline void fbfft2DVerticalCoreInverse(Complex* a) {
  // Prepare horizontal FFT
  // Twiddles is the same as for 1D but fully data parallel across threadIdx.y
  FFT1DRegisterTwiddles<FFTSize> roots(false);

  // If the input does not come in packed format, invert the unpacking done by
  // the forward pass.
  if (!PackedInput) {
    if (threadIdx.x > 0) {
      a[0] = a[0] - a[FFTSize / 2].transpose().conjugate();
    } else {
      a[0] = Complex(a[0].re(), a[FFTSize / 2].re());
    }
  }

  constexpr int HalfFFTSize = FFTSize / 2;

  // Horizontal FFT: complex FFTs as complex
#pragma unroll
  for (int i = 0; i < HalfFFTSize; ++i) {
    decimateInFrequency1DWarp<FFTSize>(a[i], roots);
  }

  // Horizontal FFT: bit reversal across threads
#pragma unroll
  for (int i = 0; i < HalfFFTSize; ++i) {
    swapHorizontal<FFTSize>(a[i]);
  }

  a[0] = a[0].conjugate() + a[0].transpose();
#pragma unroll
  for (int i = 1; i < HalfFFTSize / 2; ++i) {
    Complex Si = a[i] + a[HalfFFTSize - i].conjugate();
    Complex Di = a[i] - a[HalfFFTSize - i].conjugate();
    a[i] = Si - Di * cexp<HalfFFTSize>(i).transpose().conjugate();
    a[HalfFFTSize - i] = Si.conjugate() -
      (-Di.conjugate() * cexp<HalfFFTSize>(HalfFFTSize - i).transpose().conjugate());
  }
  Complex Si = a[HalfFFTSize / 2] + a[HalfFFTSize - HalfFFTSize / 2].conjugate();
  Complex Di = a[HalfFFTSize / 2] - a[HalfFFTSize - HalfFFTSize / 2].conjugate();
  a[HalfFFTSize / 2] = Si -
    (-Di.conjugate() * cexp<HalfFFTSize>(HalfFFTSize / 2).transpose().conjugate());

  // Vertical FFT: real FFTs as complex
  fft2dVertical<FFTSize / 2, false>(a);

  // Vertical FFT: bit reversal
  // Let the compiler unroll and optimize
#pragma unroll
  for (int i = 0; i < FFTSize / 2; ++i) {
    if (i < detail::bitReversal<FFTSize / 2>(i)) {
      // Avoid double swap
      swap(a[i], a[detail::bitReversal<FFTSize / 2>(i)]);
    }
  }
}


template <int BatchDims, int FFTSize, int BatchesPerBlock, bool Unpack = true>
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

  // Hermitian packed symmetry: Perform 2 FFTs in place
  Complex a[FFTSize / 2];
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

  // Vertical FFT first within a thread then horizontal across threads
  // Hermitian packed symmetry is used to first compute FFTSize real as
  // FTSize / 2 complex
  // Hermitian packed symmetry is further used to pack 2 real FFTs
  // (a[0] and a[FFTSize / 2 - 1]) into a single complex FFT.
  fbfft2DVerticalCoreForward<FFTSize>(a);

  // This latter symmetry needs unpacking to use with gemm routines.
  if (Unpack) {
#pragma unroll
    for (int i = 1 ; i < FFTSize / 2; ++i) {
      complexAsFloat[batch][i][threadIdx.x].template as<Complex>() = a[i];
    }

    if (threadIdx.x > 0) {
      Complex other = shfl(a[0], FFTSize - threadIdx.x, FFTSize);
      complexAsFloat[batch][0][threadIdx.x].template as<Complex>() =
        Complex(0.5f) * (other.conjugate() + a[0]);
      complexAsFloat[batch][FFTSize / 2][threadIdx.x].template as<Complex>() =
        Complex(0.5f) * (other.conjugate() - a[0]).conjugate().transpose();
    } else {
      complexAsFloat[batch][0][threadIdx.x].template as<Complex>() =
        Complex(a[0].re());
      complexAsFloat[batch][FFTSize / 2][threadIdx.x].template as<Complex>() =
        Complex(a[0].im());
    }
  } else {
    // If a specialized gemm kernel is available, no need to unpack
#pragma unroll
    for (int i = 0 ; i < FFTSize / 2; ++i) {
      complexAsFloat[batch][i][threadIdx.x].template as<Complex>() = a[i];
    }
  }
}

template <int BatchDims, int FFTSize, int BatchesPerBlock, bool PackedInput = false>
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

  // If input is not packed, read FFTSize / 2 + 1 entries and repack them later.
  constexpr int UB = (PackedInput) ? FFTSize / 2 : FFTSize / 2 + 1;

  // Perform 2 FFTs in place
  Complex a[UB];
  // Read data in
  // TODO: f16 implementation
  #pragma unroll
  for (int i = 0 ; i < UB; ++i) {
    a[i] = ldg(complexSrc[batch][i][threadIdx.x].data());
  }

  // Work it
  fbfft2DVerticalCoreInverse<FFTSize, PackedInput>(a);

  // Write the results back to memory.
  // No need for conjugation as we know we have real results.
  for (int i = 0 ; i < FFTSize; i += 2) {
    if (inBounds(i, threadIdx.x, padU, padL, realDst)) {
      realDst[batch][i - padU][threadIdx.x - padL] = a[i / 2].re();
    }
    if (inBounds(i + 1, threadIdx.x, padU, padL, realDst)) {
      realDst[batch][i + 1 - padU][threadIdx.x - padL] = a[i / 2].im();
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
__launch_bounds__(1024, 1) // 64 registers is best
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
__launch_bounds__(1024, 1) // 64 registers is best
__global__ void fbifft2DVertical_32(
    DeviceTensor<Complex, BatchDims + 2> complexSrc,
    DeviceTensor<float, BatchDims + 2> realDst,
    const int padL,
    const int padU) {
  fbifft2DVertical<BatchDims, 32, BatchesPerBlock>(
    complexSrc, realDst, padL, padU);
}

}}}}
