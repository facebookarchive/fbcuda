// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

// Julien Demouth's implementation of the Volkov strategy showed it is
// better to fully unroll by hand and have as much stuff as possible
// compile to immediate instructions.
// 2008. Volkov and Kazian, Fitting FFT onto the G80 Architecture
//
// Write our own using the same Hermitian strategy as in:
// [1412.7580] Fast Convolutional Nets With fbfft

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

#define COS_01_PI_04   0x1.6A09E6p-1f
#define COS_01_PI_08   0x1.D906BCp-1f
#define SIN_01_PI_08   0x1.87DE2Ap-2f

#define COS_00_PI_04   0x1.000000p+0f
#define COS_01_PI_04   0x1.6A09E6p-1f
#define COS_02_PI_04   0x0.000000p+0f
#define COS_03_PI_04  -0x1.6A09E6p-1f
#define SIN_00_PI_04   0x0.000000p+0f
#define SIN_01_PI_04   0x1.6A09E6p-1f
#define SIN_02_PI_04   0x1.000000p+0f
#define SIN_03_PI_04   0x1.6A09E6p-1f

#define COS_00_PI_08   0x1.000000p+0f
#define COS_01_PI_08   0x1.D906BCp-1f
#define COS_02_PI_08   0x1.6A09E6p-1f
#define COS_03_PI_08   0x1.87DE2Ap-2f
#define COS_04_PI_08   0x0.000000p+0f
#define COS_05_PI_08  -0x1.87DE2Ap-2f
#define COS_06_PI_08  -0x1.6A09E6p-1f
#define COS_07_PI_08  -0x1.D906BCp-1f
#define SIN_00_PI_08   0x0.000000p+0f
#define SIN_01_PI_08   0x1.87DE2Ap-2f
#define SIN_02_PI_08   0x1.6A09E6p-1f
#define SIN_03_PI_08   0x1.D906BCp-1f
#define SIN_04_PI_08   0x1.000000p+0f
#define SIN_05_PI_08   0x1.D906BCp-1f
#define SIN_06_PI_08   0x1.6A09E6p-1f
#define SIN_07_PI_08   0x1.87DE2Ap-2f

#define COS_00_PI_16   0x1.000000p+0f
#define COS_01_PI_16   0x1.F6297Cp-1f
#define COS_02_PI_16   0x1.D906BCp-1f
#define COS_03_PI_16   0x1.A9B662p-1f
#define COS_04_PI_16   0x1.6A09E6p-1f
#define COS_05_PI_16   0x1.1C73B4p-1f
#define COS_06_PI_16   0x1.87DE2Ap-2f
#define COS_07_PI_16   0x1.8F8B84p-3f
#define COS_08_PI_16   0x0.000000p+0f
#define COS_09_PI_16  -0x1.8F8B84p-3f
#define COS_10_PI_16  -0x1.87DE2Ap-2f
#define COS_11_PI_16  -0x1.1C73B4p-1f
#define COS_12_PI_16  -0x1.6A09E6p-1f
#define COS_13_PI_16  -0x1.A9B662p-1f
#define COS_14_PI_16  -0x1.D906BCp-1f
#define COS_15_PI_16  -0x1.F6297Cp-1f
#define SIN_00_PI_16   0x0.000000p+0f
#define SIN_01_PI_16   0x1.8F8B84p-3f
#define SIN_02_PI_16   0x1.87DE2Ap-2f
#define SIN_03_PI_16   0x1.1C73B4p-1f
#define SIN_04_PI_16   0x1.6A09E6p-1f
#define SIN_05_PI_16   0x1.A9B662p-1f
#define SIN_06_PI_16   0x1.D906BCp-1f
#define SIN_07_PI_16   0x1.F6297Cp-1f
#define SIN_08_PI_16   0x1.000000p+0f
#define SIN_09_PI_16   0x1.F6297Cp-1f
#define SIN_10_PI_16   0x1.D906BCp-1f
#define SIN_11_PI_16   0x1.A9B662p-1f
#define SIN_12_PI_16   0x1.6A09E6p-1f
#define SIN_13_PI_16   0x1.1C73B4p-1f
#define SIN_14_PI_16   0x1.87DE2Ap-2f
#define SIN_15_PI_16   0x1.8F8B84p-3f

//////////////////////////////////////////////////////////////////////////////

#define exp_01_02  Complex(            1,             0)
#define exp_01_04  Complex(            0,            -1)

#define exp_01_08  Complex( COS_01_PI_04, -COS_01_PI_04)
#define exp_02_08  Complex(            0,            -1) /* exp_01_04 */
#define exp_03_08  Complex(-COS_01_PI_04, -COS_01_PI_04)

#define exp_01_16  Complex( COS_01_PI_08, -SIN_01_PI_08)
#define exp_02_16  Complex( COS_01_PI_04, -COS_01_PI_04) /* exp_01_08 */
#define exp_03_16  Complex( SIN_01_PI_08, -COS_01_PI_08)
#define exp_04_16  Complex(            0,            -1) /* exp_01_04 */
#define exp_05_16  Complex(-SIN_01_PI_08, -COS_01_PI_08)
#define exp_06_16  Complex(-COS_01_PI_04, -COS_01_PI_04) /* exp_03_08 */
#define exp_07_16  Complex(-COS_01_PI_08, -SIN_01_PI_08)

#define exp_08_16  Complex(            1,             0) /* exp_01_02 */
#define exp_09_16  Complex(-COS_01_PI_08,  SIN_01_PI_08)

#define exp_01_32  Complex( COS_01_PI_16, -SIN_01_PI_16)
#define exp_02_32  Complex( COS_01_PI_08, -SIN_01_PI_08) /* exp_01_16 */
#define exp_03_32  Complex( COS_03_PI_16, -SIN_03_PI_16)
#define exp_04_32  Complex( COS_01_PI_04, -COS_01_PI_04) /* exp_01_08 */
#define exp_05_32  Complex( SIN_03_PI_16, -COS_03_PI_16)
#define exp_06_32  Complex( SIN_01_PI_08, -COS_01_PI_08) /* exp_03_16 */
#define exp_07_32  Complex( SIN_01_PI_16, -COS_01_PI_16)
#define exp_08_32  Complex(            0,            -1) /* exp_01_04 */
#define exp_09_32  Complex(-SIN_01_PI_16, -COS_01_PI_16)
#define exp_10_32  Complex(-SIN_01_PI_08, -COS_01_PI_08) /* exp_05_16 */
#define exp_11_32  Complex(-SIN_03_PI_16, -COS_03_PI_16)
#define exp_12_32  Complex(-COS_01_PI_04, -COS_01_PI_04) /* exp_03_08 */
#define exp_13_32  Complex(-COS_03_PI_16, -SIN_03_PI_16)
#define exp_14_32  Complex(-COS_01_PI_08, -SIN_01_PI_08) /* exp_07_16 */
#define exp_15_32  Complex(-COS_01_PI_16, -SIN_01_PI_16)

template< int > __device__ inline int rev(int);

template<> __device__ inline int rev<8>(int i)
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

template<> __device__ inline int rev<16>(int i)
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

template<> __device__ inline int rev<32>(int i)
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

static __device__ inline void FFT2(Complex &a, Complex &b)
{
  float t;
  t = a.re(); a.re() += b.re(); b.re() = t - b.re();
  t = a.im(); a.im() += b.im(); b.im() = t - b.im();
}

static __device__ inline void swap(Complex& a, Complex& b) {
  Complex t = a;
  a = b;
  b = t;
}

static __device__ inline void FFT4(
    Complex &a0, Complex &a1, Complex &a2, Complex &a3) {
  FFT2(a0, a2);
  FFT2(a1, a3);

  a3 = a3 * exp_01_04;

  FFT2(a0, a1);
  FFT2(a2, a3);
}

static __device__ inline void FFT8(Complex* a)
{
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    FFT2(a[i], a[4 + i]);
  }

  a[5] *= exp_01_08;
  a[6] *= exp_02_08;
  a[7] *= exp_03_08;

  FFT4(a[ 0], a[ 1], a[ 2], a[ 3]);
  FFT4(a[ 4], a[ 5], a[ 6], a[ 7]);
}

static __device__ inline void FFT16(Complex* a)
{
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    FFT2(a[i], a[8 + i]);
  }

  a[9]  *= exp_01_16;
  a[10] *= exp_02_16;
  a[11] *= exp_03_16;
  a[12] *= exp_04_16;
  a[13] *= exp_05_16;
  a[14] *= exp_06_16;
  a[15] *= exp_07_16;

  FFT8(a);
  FFT8(a + 8);
}


static __device__ inline void FFT32(Complex* a)
{
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    FFT2(a[i], a[16 + i]);
  }

  a[17] *= exp_01_32;
  a[18] *= exp_02_32;
  a[19] *= exp_03_32;
  a[20] *= exp_04_32;
  a[21] *= exp_05_32;
  a[22] *= exp_06_32;
  a[23] *= exp_07_32;
  a[24] *= exp_08_32;
  a[25] *= exp_09_32;
  a[26] *= exp_10_32;
  a[27] *= exp_11_32;
  a[28] *= exp_12_32;
  a[29] *= exp_13_32;
  a[30] *= exp_14_32;
  a[31] *= exp_15_32;

  FFT16(a);
  FFT16(a + 16);
}

template<int N>
static __device__ inline void fft2dCore(Complex *a);

template<> void fft2dCore<8>(Complex* a) {
  FFT8(a);
}

template<> void fft2dCore<16>(Complex* a) {
  FFT16(a);
}

template<> void fft2dCore<32>(Complex* a) {
  FFT32(a);
}

template <int FFTSize>
__device__ inline void inefficientTranspose(
    Complex* a,
    // pass shared memory buffer or lose 2x perf
    float (*sharedMem)[FFTSize][FFTSize + 1]) {

  for (int i = 0 ; i < FFTSize; ++i) {
    sharedMem[threadIdx.y][i][threadIdx.x] = a[i].re();
  }
  __syncthreads();
  for (int i = 0 ; i < FFTSize; ++i) {
    a[i].re() = sharedMem[threadIdx.y][threadIdx.x][i];
  }
  __syncthreads();
  for (int i = 0 ; i < FFTSize; ++i) {
    sharedMem[threadIdx.y][i][threadIdx.x] = a[i].im();
  }
  __syncthreads();
  for (int i = 0 ; i < FFTSize; ++i) {
    a[i].im() = sharedMem[threadIdx.y][threadIdx.x][i];
  }
}


// TODO: Partial specialization, constexpr
template <int BatchesPerBlock, int FFTSize, bool InverseFFT>
__device__ inline void inefficientTranspose(Complex* a);

///////////////////// FFT Transpose Specializations ///////////////////

// TODO: Partial specialization, constexpr
template <>
__device__ inline void inefficientTranspose<2, 32, false>(Complex* a) {
  __shared__ float sharedMem[1][32 / 2 + 1][32 + 1];

  assert(blockDim.y == 2);

  if (threadIdx.y == 0) {
    for (int i = 0 ; i < 32 / 2 + 1; ++i) {
      sharedMem[0][i][threadIdx.x] = a[i].re();
    }
    if (threadIdx.x < 32 / 2 + 1) {
      for (int i = 0 ; i < 32; ++i) {
        a[i].re() = sharedMem[0][threadIdx.x][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y == 1) {
    for (int i = 0 ; i < 32 / 2 + 1; ++i) {
      sharedMem[0][i][threadIdx.x] = a[i].re();
    }
    if (threadIdx.x < 32 / 2 + 1) {
      for (int i = 0 ; i < 32; ++i) {
        a[i].re() = sharedMem[0][threadIdx.x][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y == 0) {
    for (int i = 0 ; i < 32 / 2 + 1; ++i) {
      sharedMem[0][i][threadIdx.x] = a[i].im();
    }
    if (threadIdx.x < 32 / 2 + 1) {
      for (int i = 0 ; i < 32; ++i) {
        a[i].im() = sharedMem[0][threadIdx.x][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y == 1) {
    for (int i = 0 ; i < 32 / 2 + 1; ++i) {
      sharedMem[0][i][threadIdx.x] = a[i].im();
    }
    if (threadIdx.x < 32 / 2 + 1) {
      for (int i = 0 ; i < 32; ++i) {
        a[i].im() = sharedMem[0][threadIdx.x][i];
      }
    }
  }
}


// TODO: Partial specialization, constexpr
template <>
__device__ inline void inefficientTranspose<4, 16, false>(Complex* a) {
  __shared__ float sharedMem[4][16 / 2 + 1][16 + 1];

  for (int i = 0 ; i < 16 / 2 + 1; ++i) {
    sharedMem[threadIdx.y][i][threadIdx.x] = a[i].re();
  }
  if (threadIdx.x < 16 / 2 + 1) {
    for (int i = 0 ; i < 16; ++i) {
      a[i].re() = sharedMem[threadIdx.y][threadIdx.x][i];
    }
  }
  __syncthreads();
  for (int i = 0 ; i < 16 / 2 + 1; ++i) {
    sharedMem[threadIdx.y][i][threadIdx.x] = a[i].im();
  }
  if (threadIdx.x < 16 / 2 + 1) {
    for (int i = 0 ; i < 16; ++i) {
      a[i].im() = sharedMem[threadIdx.y][threadIdx.x][i];
    }
  }
}


// TODO: Partial specialization, constexpr
template <>
__device__ inline void inefficientTranspose<16, 8, false>(Complex* a) {
  __shared__ float sharedMem[16][8][8 + 1];

  for (int i = 0 ; i < 8 / 2 + 1; ++i) {
    sharedMem[threadIdx.y][i][threadIdx.x] = a[i].re();
  }
  if (threadIdx.x < 8 / 2 + 1) {
    for (int i = 0 ; i < 8; ++i) {
      a[i].re() = sharedMem[threadIdx.y][threadIdx.x][i];
    }
  }
  __syncthreads();
  for (int i = 0 ; i < 8 / 2 + 1; ++i) {
    sharedMem[threadIdx.y][i][threadIdx.x] = a[i].im();
  }
  if (threadIdx.x < 8 / 2 + 1) {
    for (int i = 0 ; i < 8; ++i) {
      a[i].im() = sharedMem[threadIdx.y][threadIdx.x][i];
    }
  }
}

///////////////////// IFFT Transpose Specializations ///////////////////

// TODO: Partial specialization, constexpr
template <>
__device__ inline void inefficientTranspose<2, 32, true>(Complex* a) {
  __shared__ float sharedMem[1][32][32 + 1];

  assert(blockDim.y == 2);

  if (threadIdx.y == 0) {
    for (int i = 0 ; i < 32; ++i) {
      sharedMem[0][i][threadIdx.x] = a[i].re();
    }
    if (threadIdx.x < 32) {
      for (int i = 0 ; i < 32; ++i) {
        a[i].re() = sharedMem[0][threadIdx.x][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y == 1) {
    for (int i = 0 ; i < 32; ++i) {
      sharedMem[0][i][threadIdx.x] = a[i].re();
    }
    if (threadIdx.x < 32) {
      for (int i = 0 ; i < 32; ++i) {
        a[i].re() = sharedMem[0][threadIdx.x][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y == 0) {
    for (int i = 0 ; i < 32; ++i) {
      sharedMem[0][i][threadIdx.x] = a[i].im();
    }
    if (threadIdx.x < 32) {
      for (int i = 0 ; i < 32; ++i) {
        a[i].im() = sharedMem[0][threadIdx.x][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y == 1) {
    for (int i = 0 ; i < 32; ++i) {
      sharedMem[0][i][threadIdx.x] = a[i].im();
    }
    if (threadIdx.x < 32) {
      for (int i = 0 ; i < 32; ++i) {
        a[i].im() = sharedMem[0][threadIdx.x][i];
      }
    }
  }
}


// TODO: Partial specialization, constexpr
template <>
  __device__ inline void inefficientTranspose<4, 16, true>(Complex* a) {
  __shared__ float sharedMem[4][16][16 + 1];

  for (int i = 0 ; i < 16; ++i) {
    sharedMem[threadIdx.y][i][threadIdx.x] = a[i].re();
  }
  for (int i = 0 ; i < 16; ++i) {
    a[i].re() = sharedMem[threadIdx.y][threadIdx.x][i];
  }
  __syncthreads();
  for (int i = 0 ; i < 16; ++i) {
    sharedMem[threadIdx.y][i][threadIdx.x] = a[i].im();
  }
  for (int i = 0 ; i < 16; ++i) {
    a[i].im() = sharedMem[threadIdx.y][threadIdx.x][i];
  }
}

// TODO: Partial specialization, constexpr
template <>
__device__ inline void inefficientTranspose<16, 8, true>(Complex* a) {
  __shared__ float sharedMem[16][8][8 + 1];

  for (int i = 0 ; i < 8; ++i) {
    sharedMem[threadIdx.y][i][threadIdx.x] = a[i].re();
  }
  for (int i = 0 ; i < 8; ++i) {
    a[i].re() = sharedMem[threadIdx.y][threadIdx.x][i];
  }
  __syncthreads();
  for (int i = 0 ; i < 8; ++i) {
    sharedMem[threadIdx.y][i][threadIdx.x] = a[i].im();
  }
  for (int i = 0 ; i < 8; ++i) {
    a[i].im() = sharedMem[threadIdx.y][threadIdx.x][i];
  }
}


//////////////////////////// FBFFT Generic ////////////////////////////////

template <int FFTSize, int BatchesPerBlock, bool InverseFFT>
__device__ __forceinline__ void fbfft2DCore(Complex* a) {
  // B. 2 real FFTs as one complex
  fft2dCore<FFTSize>(a);

  // C. Bit reversal
  // Let the compiler unroll and optimize
#pragma unroll
  for (int i = 0; i < FFTSize; ++i) {
    if (i < detail::rev<FFTSize>(i)) {
      // Avoid double swap
      swap(a[i], a[detail::rev<FFTSize>(i)]);
    }
  }

  // D. Inefficiently transpose
  inefficientTranspose<BatchesPerBlock, FFTSize, InverseFFT> (a);

  // Use Hermitian symmetry
  // Almost 1/2 threads do nothing here, we don't care we're memory bound
  // because we're register bound.
  if (InverseFFT || threadIdx.x < FFTSize / 2 + 1) {
    // E. FFT the rows
    fft2dCore<FFTSize>(a);

    // F. Bit reversal
    // Let the compiler unroll and optimize
#pragma unroll
    for (int i = 0; i < FFTSize; ++i) {
      if (i < detail::rev<FFTSize>(i)) {
        // Avoid double swap
        swap(a[i], a[detail::rev<FFTSize>(i)]);
      }
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
__global__ void fbfft2D(
    DeviceTensor<float, BatchDims + 2> real,
    DeviceTensor<float, BatchDims + 3> complexAsFloat,
    const int padL,
    const int padU) {
  assert(BatchesPerBlock >= 1);
  assert(gridDim.x == 1);
  assert(gridDim.z == 1);
  assert(blockDim.z == 1);

  const int batch =
    BatchesPerBlock * (blockIdx.x * gridDim.y + blockIdx.y ) + threadIdx.y;

  // Early exit if we would overflow
  if (batch >= real.getSize(0)) {
    return;
  }

  // Perform 2 FFTs in place
  Complex a[FFTSize];
  // A. read data in
  // TODO: read as float2
  // TODO: f16 implementation
  for (int i = 0 ; i < FFTSize; ++i) {
    a[i] = inBounds(i, threadIdx.x, padU, padL, real) ?
      Complex(real[batch][i - padU][threadIdx.x - padL].ldg()) :
      Complex(0.0f, 0.0f);
  }

  // B. - F.
  fbfft2DCore<FFTSize, BatchesPerBlock, false>(a);

  // G. Write output in pieces, using symmetry
  if (threadIdx.x < FFTSize / 2 + 1) {
    // 1. Write [0 , FFTSize / 2 + 1) ^ 2
#pragma unroll
    for (int i = 0 ; i < FFTSize / 2 + 1; ++i) {
      complexAsFloat[batch][i][threadIdx.x].template as<Complex>() = a[i];
    }

    if (0 < threadIdx.x  && threadIdx.x < FFTSize / 2) {
      {
        // 2. Orthogonal symmetry for i == 0
        int i = 0;
        complexAsFloat[batch][i][FFTSize - threadIdx.x].
          template as<Complex>() = a[i].conjugate();
      }

      // 3. Central symmetry for:
      // [FFTSize / 2 + 1, FFTSize) x [1, FFTSize / 2)
#pragma unroll
      for (int i = 1; i < FFTSize / 2; ++i) {
        complexAsFloat[batch][i][FFTSize - threadIdx.x].
          template as<Complex>() = a[FFTSize - i].conjugate();
      }

      {
        // 4. Orthogonal symmetry for i == FFTSize / 2
        int i = FFTSize / 2;
        complexAsFloat[batch][i][FFTSize - threadIdx.x].
          template as<Complex>() = a[i].conjugate();
      }
    }
  }
}

template <int BatchDims, int FFTSize, int BatchesPerBlock>
__global__ void fbifft2D(
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
  for (int i = 0 ; i < FFTSize / 2 + 1; ++i) {
    a[i] = complexSrc[batch][i][threadIdx.x].data()->conjugate();
  }
  if (threadIdx.x == 0 || threadIdx.x == FFTSize / 2) {
    // 2. Orthogonal symmetry for first and middle columns along horizontal
    // plane FFTSize / 2 = 1
    for (int i = FFTSize / 2 + 1; i < FFTSize; ++i) {
      a[i] = a[FFTSize - i].conjugate();
    }
  } else {
    // 3. Central symmetry for:
    //   [1, FFTSize / 2) x [FFTSize / 2 + 1, FFTSize) and
    //   [FFTSize / 2 + 1, FFTSize) x [FFTSize / 2 + 1, FFTSize)
    for (int i = FFTSize / 2 + 1; i < FFTSize; ++i) {
      // conjugate().conjugate() == id
      a[i] = complexSrc[batch][FFTSize - i][FFTSize - threadIdx.x];
    }
  }

  // B. - F.
  fbfft2DCore<FFTSize, BatchesPerBlock, true>(a);

  // C. Write the results back to memory.
  // No need for conjugation as we know we have real results.
  for (int i = 0 ; i < FFTSize; ++i) {
    if (inBounds(i, threadIdx.x, padU, padL, realDst)) {
      realDst[batch][i - padU][threadIdx.x - padL] = a[i].re();
    }
  }
}

}}}}
