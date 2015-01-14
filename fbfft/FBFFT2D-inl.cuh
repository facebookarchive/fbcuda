// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/Complex.cuh"
#include "cuda/ComputeCapabilities.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/fbfft/FBFFTCommon.cuh"

#include <cuda_runtime.h>
#include <glog/logging.h>

using namespace facebook::cuda;

namespace facebook { namespace cuda { namespace fbfft {

namespace detail {

template <int FFTSize>
__device__ __forceinline__ void load2D(
    const DeviceTensor<float, 3>& real,
    FFT1DCoeffs<FFTSize>& coeffs,
    const int batch,
    const int indexX,
    const int indexY) {
  int LogFFTSize = getMSB<FFTSize>();
  // adjustedThreadIdxX<FFTSize>() crams multiple < WARP_SIZE FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + indexX * blockDim.x;
  // adjustedThreadIdxX<FFTSize>() crams multiple < WARP_SIZE FFTs in a warp
  int y = adjustedThreadIdxY<FFTSize>() + indexY * blockDim.y;

  // Zero padding without a need to copy the input data to a larger array.
  coeffs[indexX] =
    Complex((y < real.getSize(1) && x < real.getSize(2)) ?
            real[batch][y][x].ldg() : 0.0f,
            0.0f);
}

template <int FFTSize, bool EvenDivideBatches>
__device__ __forceinline__ void load2DR2C(
    const DeviceTensor<float, 3>& real,
    FFT1DCoeffs<FFTSize>& coeffs,
    const int batch,
    const int indexX,
    const int indexY) {
  int LogFFTSize = getMSB<FFTSize>();
  // adjustedThreadIdxX<FFTSize>() crams multiple < WARP_SIZE FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + indexX * blockDim.x;
  // adjustedThreadIdxX<FFTSize>() crams multiple < WARP_SIZE FFTs in a warp
  int y = adjustedThreadIdxY<FFTSize>() + indexY * blockDim.y;

  // Zero padding without a need to copy the input data to a larger array.
  coeffs[indexX] = (y < real.getSize(1) && x < real.getSize(2)) ?
    Complex(real[batch][y][x].ldg(),
            (EvenDivideBatches || batch + 1 < real.getSize(0)) ?
            real[batch + 1][y][x].ldg()
            :
            0.0f)
    :
    Complex(0.0f);
}

template <int FFTSize>
__device__ __forceinline__ void store2D(
    DeviceTensor<float, 4>& complexAsFloat,
    const FFT1DCoeffs<FFTSize>& coeffs,
    const int batch,
    const int indexX,
    const int indexY) {
  // adjustedThreadIdxX<FFTSize>() crams multiple < WARP_SIZE FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + indexX * blockDim.x;
  // adjustedThreadIdxX<FFTSize>() crams multiple < WARP_SIZE FFTs in a warp
  int y = adjustedThreadIdxY<FFTSize>() + indexY * blockDim.y;
  if (y < complexAsFloat.getSize(1) && x < complexAsFloat.getSize(2)) {
    complexAsFloat[batch][y][x][0].as<Complex>() = coeffs[indexX];
  }
}

// Performs cross warp transpose of the data in registers, synchronously for
// each register at a time and takes advantage of Hermitian symmetry.
//
// Supports multiple FFTs per warp.
//
// Invariants are:
//  - not synchronized on entry of the loop
//  - synchronized at each step of the loop
//  - synchronized on exit
template <int FFTSize, int SMemRows, int RowsPerWarp, int FFTPerWarp>
__device__ __forceinline__ void transpose2DHermitianMultiple(
      FFT1DCoeffs<FFTSize> (&coeffsArray)[RowsPerWarp],
      Complex(*buffer)[SMemRows + 1]) {
  const int LogFFTSize = getMSB<FFTSize>();
  const int thx0 = (threadIdx.x >> LogFFTSize) << LogFFTSize;
#pragma unroll
  for (int row = 0; row < RowsPerWarp; ++row) {
    int y = adjustedThreadIdxY<FFTSize>() + row * blockDim.y;
    FFT1DCoeffs<FFTSize>& coeffs = coeffsArray[row];
    buffer[y][threadIdx.x] = coeffs.coeff[0];
  }

  __syncthreads();

#pragma unroll
  for (int row = 0; row < RowsPerWarp; ++row) {
    int y = adjustedThreadIdxY<FFTSize>() + row * blockDim.y;
    FFT1DCoeffs<FFTSize>& coeffs = coeffsArray[row];
    coeffs.coeff[0] =
      buffer
      [adjustedThreadIdxX<FFTSize>()]
      [thx0 + y];
  }

  __syncthreads();
}

template <int FFTSize, int FFTPerWarp, int RowsPerWarp, bool BitReverse>
__global__ void decimateInFrequencyHermitian2DWarpKernel(
    DeviceTensor<float, 3> real, DeviceTensor<float, 4> complexAsFloat) {
  cuda_static_assert(!(FFTPerWarp & (FFTPerWarp - 1)));
  cuda_static_assert(FFTPerWarp * FFTSize <= WARP_SIZE);
  // Only let FFTs <= 8 have multiple per warp, 16 and 32 perform better with
  // 1 per warp.
  cuda_static_assert(FFTSize <= WARP_SIZE);
  assert(FFTPerWarp * FFTSize == blockDim.x);
  assert(real.getSize(0) % FFTPerWarp == 0);

  int LogFFTSize = getMSB<FFTSize>();
  // Enforce that the number of FFTs we perform is divisible by the number of
  // FFTs per warp, otherwise weird divergence will occur and possibly bugs.
  const int batch = adjustedBatchR2C<FFTSize, FFTPerWarp, true>();
  if (batch >= real.getSize(0)) {
    return;
  }

  // Can support multiple rows of FFT per warp if needed, atm use 1
  FFT1DCoeffs<FFTSize> coeffsArray[RowsPerWarp];
  FFT1DCoeffs<FFTSize> coeffsArray2[2][RowsPerWarp];
#pragma unroll
  for (int i = 0; i < RowsPerWarp; ++i) {
    load2DR2C<FFTSize, false>(real, coeffsArray[i], batch, 0, i);
  }

  // Twiddles is the same as for 1D but fully data parallel across threadIdx.y
  FFT1DRoots<FFTSize> roots;
  roots.template twiddles<true>();

#pragma unroll
  for (int i = 0; i < RowsPerWarp; ++i) {
    decimateInFrequency1DWarp<FFTSize>(coeffsArray[i][0], roots[0]);
  }
  FFT1DBitReversal<FFTSize> bits;
  if (BitReverse) {
    bits.computeBitReversal(0);
#pragma unroll
    for (int i = 0; i < RowsPerWarp; ++i) {
      bitReverse1DWarp<FFTSize, FFTPerWarp>(coeffsArray[i], bits, 0);
    }
  }

#pragma unroll
  for (int i = 0; i < RowsPerWarp; ++i) {
    Complex other =
      shfl(coeffsArray[i][0],
           FFTSize - adjustedThreadIdxX<FFTSize>(),
           FFTSize);
    coeffsArray2[0][i].coeff[0] =
      Complex(0.5f * (coeffsArray[i][0].re() + other.re()),
              0.5f * (coeffsArray[i][0].im() - other.im()));
    coeffsArray2[1][i].coeff[0] =
      Complex(0.5f * ( coeffsArray[i][0].im() + other.im()),
              0.5f * (-coeffsArray[i][0].re() + other.re()));
  }

  __shared__ Complex buffer[FFTSize][WARP_SIZE + 1];
  transpose2DHermitianMultiple<FFTSize, WARP_SIZE, RowsPerWarp, FFTPerWarp>(
    coeffsArray2[0],
    (Complex(*)[WARP_SIZE + 1])buffer);
  transpose2DHermitianMultiple<FFTSize, WARP_SIZE, RowsPerWarp, FFTPerWarp>(
    coeffsArray2[1],
    (Complex(*)[WARP_SIZE + 1])buffer);

#pragma unroll
  for (int i = 0; i < RowsPerWarp; ++i) {
    int y = adjustedThreadIdxY<FFTSize>() + i * blockDim.y;
    if (y < FFTSize / 2 + 1) {
      decimateInFrequency1DWarp<FFTSize>(coeffsArray2[0][i][0], roots[0]);
      decimateInFrequency1DWarp<FFTSize>(coeffsArray2[1][i][0], roots[0]);
    }
  }

  if (BitReverse) {
#pragma unroll
    for (int i = 0; i < RowsPerWarp; ++i) {
      int y = adjustedThreadIdxY<FFTSize>() + i * blockDim.y;
      if (y < FFTSize / 2 + 1) {
        // Bit reversal is the same as for 1D but fully data parallel across
        // threadIdx.y
        bitReverse1DWarp<FFTSize, FFTPerWarp>(coeffsArray2[0][i], bits, 0);
        bitReverse1DWarp<FFTSize, FFTPerWarp>(coeffsArray2[1][i], bits, 0);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < RowsPerWarp; ++i) {
    store2D<FFTSize>(complexAsFloat, coeffsArray2[0][i], batch, 0, i);
    if (batch + 1 < real.getSize(0)) {
      store2D<FFTSize>(complexAsFloat, coeffsArray2[1][i], batch + 1, 0, i);
    }
  }
}


// First half of the 2-D transform for >= 64.
//
// This is a good 2D kernel, with 64, 1, 4, 4 sizing and the configuration
// below it is 10% faster than the equivalent batched 1-D version, even if it
// has only 1/2 the occupancy.
template <int FFTSize, int RowsPerKernel, int BlockDimY, bool BitReverse>
__launch_bounds__(32 * 8, 4) // 64 X 64 and 128 x 128
__global__ void decimateInFrequency2DKernel(
    DeviceTensor<float, 3> real,
    DeviceTensor<float, 4> complexAsFloat) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BlockDimY);
  assert(real.getSize(0) == complexAsFloat.getSize(0));
  // This version does not deal with a whole N x N FFT within a single block.
  // It *cannot* update in place transposed -> ensure we have the same
  // dimensions to update one row at a time.

  int LogFFTSize = getMSB<FFTSize>();
  // Enforce that the number of FFTs we perform is divisible by the number of
  // FFTs per warp, otherwise weird divergence will occur and possibly bugs
  const int batch = adjustedBatch<FFTSize, 1>();
  if (batch >= real.getSize(0)) {
    return;
  }

  for (int yiter = 0; yiter < FFTSize; yiter += RowsPerKernel * BlockDimY) {
    FFT1DCoeffs<FFTSize> coeffsArray[RowsPerKernel];
    const int ColumnsPerWarp = coeffsArray[0].ColumnsPerWarp;

    __shared__ Complex buffer[BlockDimY][FFTSize];

#pragma unroll
    for (int row = 0; row < RowsPerKernel; ++row) {
      int y = yiter + threadIdx.y + row * blockDim.y;
#pragma unroll
      for (int reg = 0; reg < ColumnsPerWarp; ++reg) {
        int x = threadIdx.x + reg * blockDim.x;
        coeffsArray[row][reg] =
          Complex((y < real.getSize(1) && x < real.getSize(2)) ?
                  real[batch][y][x].ldg() : 0.0f,
                  0.0f);
      }
    }

    {
      // Twiddles is the same as for 1D but fully data parallel on threadIdx.y
      FFT1DRoots<FFTSize> roots;
      roots.template twiddles<true>();

      decimateInFrequency1D<FFTSize, 1, RowsPerKernel, 0, RowsPerKernel>(
        coeffsArray, roots, batch);
    }

    if (BitReverse) {
      FFT1DBitReversal<FFTSize> bits;
#pragma unroll
      for (int reg = 0; reg < ColumnsPerWarp; ++reg) {
        bits.computeBitReversal(reg);
      }

      Complex (*buffer2) [FFTSize] = (Complex(*)[FFTSize])buffer;
      // bitReverse all
#pragma unroll
      for (int row = 0; row < RowsPerKernel; ++row) {
#pragma unroll
        for (int reg = 0; reg < ColumnsPerWarp; ++reg) {
          int x = getLaneId() + reg * WARP_SIZE;
          buffer2[threadIdx.y][x] = coeffsArray[row][reg];
        }
#pragma unroll
        for (int reg = 0; reg < ColumnsPerWarp; ++reg) {
          coeffsArray[row][reg] = buffer2[threadIdx.y][bits[reg]];
        }
      }
      // No need to sync up here, no following kernel
    }

#pragma unroll
    for (int row = 0; row < RowsPerKernel; ++row) {
      int y = yiter + threadIdx.y + row * blockDim.y;
#pragma unroll
      for (int reg = 0; reg < ColumnsPerWarp; ++reg) {
        int x = threadIdx.x + reg * blockDim.x;
        if (y < complexAsFloat.getSize(1) && x < complexAsFloat.getSize(2)) {
          *(complexAsFloat[batch][y][x].dataAs<Complex>()) =
            coeffsArray[row][reg];
        }
      }
    }
  }
}

// Second half of the 2-D transform for >= 64.
//
template <int FFTSize, int RowsPerKernel, int BlockDimY, bool BitReverse>
__device__ __forceinline__ void decimateInFrequency2DKernel(
    const DeviceTensor<Complex, 3> src,
    DeviceTensor<Complex, 3> dst) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BlockDimY);
  assert(src.getSize(0) == dst.getSize(0));
  // This version does not deal with a whole N x N FFT within a single block.
  // It *cannot* update in place transposed -> ensure we are writing to 2
  // different storage areas
  assert(src.data() != dst.data());

  int LogFFTSize = getMSB<FFTSize>();
  // Enforce that the number of FFTs we perform is divisible by the number of
  // FFTs per warp, otherwise weird divergence will occur and possibly bugs
  const int batch = adjustedBatch<FFTSize, 1>();
  if (batch >= src.getSize(0)) {
    return;
  }

  const int UpperBound = FFTSize / 2 + 1;
  for (int yiter = 0;
       yiter < UpperBound;
       yiter += RowsPerKernel * BlockDimY) {
    // Split into lower and upper half, upper half will be cut by symmetry
    FFT1DCoeffs<FFTSize> coeffsArray[RowsPerKernel];
    const int ColumnsPerWarp = coeffsArray[0].ColumnsPerWarp;

    __shared__ Complex buffer[BlockDimY][FFTSize];

#pragma unroll
    for (int row = 0; row < RowsPerKernel; ++row) {
      int y = yiter + threadIdx.y + row * blockDim.y;
#pragma unroll
      for (int reg = 0; reg < ColumnsPerWarp; ++reg) {
        int x = threadIdx.x + reg * blockDim.x;
        // This is the key: uncoalesced, transposed reads using ldg work
        // really well and remove the need for doing an actual transpose.
        // TODO: Awkward ldg use but does the job
        coeffsArray[row][reg] = (x < src.getSize(1) && y < src.getSize(2)) ?
          ldg(src[batch][x][y].data()) : Complex(0.0f);
      }
    }

    {
      // Twiddles is the same as for 1D but fully data parallel wrt threadIdx.y
      FFT1DRoots<FFTSize> roots;
      roots.template twiddles<true>();

      decimateInFrequency1D<FFTSize,
                            1,
                            RowsPerKernel,
                            0,
                            RowsPerKernel>(coeffsArray, roots, batch);
    }

    if (BitReverse) {
      {
        FFT1DBitReversal<FFTSize> bits;
#pragma unroll
        for (int reg = 0; reg < ColumnsPerWarp; ++reg) {
          bits.computeBitReversal(reg);
        }

        Complex (*buffer2) [FFTSize] = (Complex(*)[FFTSize])buffer;
        // bitReverse all
#pragma unroll
        for (int row = 0; row < RowsPerKernel; ++row) {
#pragma unroll
          for (int reg = 0; reg < ColumnsPerWarp; ++reg) {
            int x = getLaneId() + reg * WARP_SIZE;
            buffer2[threadIdx.y][x] = coeffsArray[row][reg];
          }
#pragma unroll
          for (int reg = 0; reg < ColumnsPerWarp; ++reg) {
            coeffsArray[row][reg] = buffer2[threadIdx.y][bits[reg]];
          }
        }
        // No need to sync up here, no following smem access
      }
    }

    // If needed, could reintroduce the "untranspose" feature but this is
    // expensive for sizes > 32

#pragma unroll
    for (int row = 0; row < RowsPerKernel; ++row) {
      int y = yiter + threadIdx.y + row * blockDim.y;
#pragma unroll
      for (int reg = 0; reg < ColumnsPerWarp; ++reg) {
        int x = threadIdx.x + reg * blockDim.x;
        if (y < dst.getSize(1) && x < dst.getSize(2)) {
          dst[batch][y][x] = coeffsArray[row][reg];
        }
      }
    }
  }
}



template <int FFTSize, int RowsPerKernel, int BlockDimY, bool BitReverse,
          bool ForwardFFT>
 __launch_bounds__(32 * 32, 1)
__global__ void decimateInFrequency2DKernel128(
    const DeviceTensor<Complex, 3> src,
    DeviceTensor<Complex, 3> dst) {
  decimateInFrequency2DKernel<FFTSize,
                              RowsPerKernel,
                              BlockDimY,
                              BitReverse>(src, dst);
}

template <int FFTSize, int RowsPerKernel, int BlockDimY, bool BitReverse,
          bool ForwardFFT>
__launch_bounds__(32 * 32, 2)
  __global__ void decimateInFrequency2DKernel64(
    const DeviceTensor<Complex, 3> src,
    DeviceTensor<Complex, 3> dst) {
  decimateInFrequency2DKernel<FFTSize,
                              RowsPerKernel,
                              BlockDimY,
                              BitReverse>(src, dst);
}

} // namespace

// First half of the forward 2-D transform
// Only transform to be called for <= 32
template <int BatchDims>
FBFFTParameters::ErrorCode fbfft2D(
    DeviceTensor<float, BatchDims + 2>& real,
    DeviceTensor<float, BatchDims + 3>& complexAsFloat,
    cudaStream_t s) {
  initTwiddles();

  // TODO: The limiter for size 256 is the twiddle cross-register shuffle
  // implementation that is currently unrolled by hand.
  // TODO: Starting 512, the occupancy goes down due to shared memory bit
  // reversal.
  // Whatever the real input size, we can make assumptions on the
  // complexAsFloat size related to the fft size (because interpolation).
  // If buffer, it must be sized N x (N / 2 +1)
  assert(complexAsFloat.getSize(2) <= 32 ||
         (complexAsFloat.getSize(2) ==
          numHermitian(complexAsFloat.getSize(1))));
  // If buffer, it must be sized (N / 2 + 1) x N
  assert(complexAsFloat.getSize(1) > 32 ||
         (complexAsFloat.getSize(1) ==
          numHermitian(complexAsFloat.getSize(2))));
  if (complexAsFloat.getSize(1) > 256) {
    return FBFFTParameters::UnsupportedSize;
  }

#define SELECT_FBFFT_2D_DIF_WARP_SINGLE(                                \
  FFT_SIZE, FFTS_PER_WARP, BIT_REVERSE, ROWS_PER_WARP)                  \
  cuda_static_assert(FFT_SIZE <= WARP_SIZE);                            \
  if (complexAsFloat.getSize(2) == FFT_SIZE) {                          \
    if (real.getSize(0) % (2 * FFTS_PER_WARP) == 0) {                   \
      dim3 blocks(ceil(real.getSize(0), 2 * FFTS_PER_WARP));            \
      dim3 threads(FFT_SIZE * FFTS_PER_WARP,                            \
                   ceil(FFT_SIZE, ROWS_PER_WARP));                      \
      detail::decimateInFrequencyHermitian2DWarpKernel<                 \
        FFT_SIZE, FFTS_PER_WARP, ROWS_PER_WARP, BIT_REVERSE>            \
        <<<blocks, threads, 0, s>>>(real, complexAsFloat);              \
    } else {                                                            \
      dim3 blocks(ceil(complexAsFloat.getSize(0), 2));                  \
      dim3 threads(FFT_SIZE, FFT_SIZE);                                 \
      detail::decimateInFrequencyHermitian2DWarpKernel<                 \
        FFT_SIZE, 1, 1, BIT_REVERSE>                                    \
        <<<blocks, threads, 0, s>>>(real, complexAsFloat);              \
    }                                                                   \
    return FBFFTParameters::Success;                                    \
  }

  // Above warp level, buffer is needed, output must be N x (N / 2 + 1)
#define SELECT_FBFFT_2D_DIF_SINGLE(                                     \
  FFT_SIZE, ROWS_PER_KERNEL, BLOCKDIMY, BIT_REVERSE)                    \
  if (complexAsFloat.getSize(1) == FFT_SIZE) {                          \
  dim3 blocks(complexAsFloat.getSize(0));                               \
  dim3 threads(WARP_SIZE, BLOCKDIMY);                                   \
  detail::decimateInFrequency2DKernel<                                  \
    FFT_SIZE,  ROWS_PER_KERNEL, BLOCKDIMY, BIT_REVERSE>                 \
    <<<blocks, threads, 0, s>>>(real, complexAsFloat);                  \
  return FBFFTParameters::Success;                                      \
}

  SELECT_FBFFT_2D_DIF_WARP_SINGLE( 2, 16, true, 1);
  SELECT_FBFFT_2D_DIF_WARP_SINGLE( 4,  8, true, 1);
  SELECT_FBFFT_2D_DIF_WARP_SINGLE( 8,  4, true, 2);
  SELECT_FBFFT_2D_DIF_WARP_SINGLE(16,  2, true, 4);
  SELECT_FBFFT_2D_DIF_WARP_SINGLE(32,  1, true, 4);
  SELECT_FBFFT_2D_DIF_SINGLE( 64, 4, 4, true);
  SELECT_FBFFT_2D_DIF_SINGLE(128, 4, 4, true);

#undef SELECT_FBFFT_2D_DIF_WARP_SINGLE
#undef SELECT_FBFFT_2D_DIF_SINGLE

  return FBFFTParameters::UnsupportedSize;
}


// Second half of the 2-D transform for >= 64
template <int BatchDims>
FBFFTParameters::ErrorCode fbfft2D(
    DeviceTensor<Complex, BatchDims + 2>& complexSrc,
    DeviceTensor<Complex, BatchDims + 2>& complexDst,
    cudaStream_t s) {

  initTwiddles();

  // TODO: The limiter for size 256 is the twiddle cross-register shuffle
  // implementation that is currently unrolled by hand.
  // TODO: Starting 512, the occupancy goes down due to shared memory bit
  // reversal.
  // Input is the temporary buffer and must be sized as N x (N / 2 + 1)
  assert((complexSrc.getSize(2) ==
          numHermitian(complexSrc.getSize(1))));
  // If we are here we must be >= 64
  assert(complexSrc.getSize(1) >= 64);
  // Output is the real output and must be sized as the input, must enforce
  // this upstream
  if (complexSrc.getSize(2) > 256) {
    return FBFFTParameters::UnsupportedSize;
  }

#define SELECT_FBFFT_2D_DIF_SINGLE(                                     \
  FFT_SIZE, ROWS_PER_KERNEL, BLOCKDIMY, BIT_REVERSE)                    \
  if (complexSrc.getSize(1) == FFT_SIZE) {                              \
    dim3 blocks(complexSrc.getSize(0));                                 \
    dim3 threads(32, BLOCKDIMY);                                        \
    detail::decimateInFrequency2DKernel##FFT_SIZE<                      \
      FFT_SIZE,  ROWS_PER_KERNEL, BLOCKDIMY, BIT_REVERSE, true>         \
      <<<blocks, threads, 0, s>>>(complexSrc, complexDst);              \
      return FBFFTParameters::Success;                                  \
  }

  SELECT_FBFFT_2D_DIF_SINGLE(64, 2, 17, true);
  SELECT_FBFFT_2D_DIF_SINGLE(128, 1, 17, true);

#undef SELECT_FBFFT_2D_DIF_SINGLE

  return FBFFTParameters::UnsupportedSize;
}

} } } // namespace
