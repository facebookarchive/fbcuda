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

template <int BatchDims>
FBFFTParameters::ErrorCode fbfft1D(
    DeviceTensor<float, BatchDims + 1>& real,
    DeviceTensor<float, BatchDims + 2>& complex,
    cudaStream_t s) {

  initTwiddles();

  // TODO: The limiter for size 256 is the twiddle cross-register shuffle
  // implementation that is currently unrolled by hand.
  // TODO: Starting 512, the occupancy goes down due to shared memory bit
  // reversal.
  assert(real.getSize(1) <= 256);
  assert(BatchDims == 1);

  // TODO: this drops to 1 FFT per warp if batch size is not an even multiple
  // of FFTS_PER_WARP -> implement kernel and epilogue to handle most cases
  // efficiently.
#define SELECT_FBFFT_1D_DIF_LE_WARP_SIZE(FFT_SIZE, BATCH_UNROLL, FFTS_PER_WARP) \
  if (real.getSize(1) == FFT_SIZE) {                                    \
    cuda_static_assert(FFT_SIZE <= WARP_SIZE);                          \
    if (real.getSize(0) % (2 * FFTS_PER_WARP * BATCH_UNROLL) == 0) {    \
      dim3 blocks(ceil(ceil(real.getSize(0), 2 * FFTS_PER_WARP),        \
                       BATCH_UNROLL));                                  \
      dim3 threads(real.getSize(1) * FFTS_PER_WARP, 1, BATCH_UNROLL);   \
      detail::decimateInFrequency1DKernel<FFT_SIZE, 1, FFTS_PER_WARP, true> \
        <<<blocks, threads, 0, s>>>(real, complex);                     \
      return FBFFTParameters::Success;                                  \
    } else if (BATCH_UNROLL == 1 ||                                     \
               real.getSize(0) % (2 * BATCH_UNROLL) == 0) {             \
      dim3 blocks(ceil(real.getSize(0), 2 * BATCH_UNROLL));             \
      dim3 threads(real.getSize(1), 1, BATCH_UNROLL);                   \
      detail::decimateInFrequency1DKernel<FFT_SIZE, 1, 1, true>         \
        <<<blocks, threads, 0, s>>>(                                    \
          real, complex);                                               \
      return FBFFTParameters::Success;                                  \
    }                                                                   \
  }

#define SELECT_FBFFT_1D_DIF_GT_WARP_SIZE(FFT_SIZE, BATCH_UNROLL)        \
  if (real.getSize(1) == FFT_SIZE) {                                    \
  cuda_static_assert(FFT_SIZE > WARP_SIZE);                             \
  dim3 blocks(ceil(real.getSize(0), (2 * BATCH_UNROLL)));               \
  dim3 threads(WARP_SIZE, 1, BATCH_UNROLL);                             \
  detail::decimateInFrequency1DKernel<FFT_SIZE, BATCH_UNROLL, 1, true>  \
    <<<blocks, threads, 0, s>>>(real, complex);                         \
  return FBFFTParameters::Success;                                      \
}

  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE( 2, 32, 16);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE( 2,  1, 16);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE( 4, 16,  8);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE( 4,  1,  8);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE( 8,  8,  4);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE( 8,  1,  4);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE(16,  4,  2);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE(16,  1,  2);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE(32,  4,  1);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE(32,  1,  1);
  SELECT_FBFFT_1D_DIF_GT_WARP_SIZE(64,  4);
  SELECT_FBFFT_1D_DIF_GT_WARP_SIZE(128, 4);
  SELECT_FBFFT_1D_DIF_GT_WARP_SIZE(256, 2);

#undef SELECT_FBFFT_1D_DIF_LE_WARP_SIZE
#undef SELECT_FBFFT_1D_DIF_GT_WARP_SIZE

  return FBFFTParameters::UnsupportedSize;
}

template <int BatchDims>
FBFFTParameters::ErrorCode fbifft1D(
    DeviceTensor<float, BatchDims + 1>& real,
    DeviceTensor<float, BatchDims + 2>& complex,
    cudaStream_t s) {

  initTwiddles();

  // TODO: The limiter for size 256 is the twiddle cross-register shuffle
  // implementation that is currently unrolled by hand.
  // TODO: Starting 512, the occupancy goes down due to shared memory bit
  // reversal.
  assert(real.getSize(1) <= 256);
  assert(BatchDims == 1);

  // TODO: this drops to 1 FFT per warp if batch size is not an even multiple
  // of FFTS_PER_WARP -> implement kernel and epilogue to handle most cases
  // efficiently.
#define SELECT_FBFFT_1D_DIF_LE_WARP_SIZE(FFT_SIZE, BATCH_UNROLL, FFTS_PER_WARP) \
  if (real.getSize(1) == FFT_SIZE) {                                    \
    cuda_static_assert(FFT_SIZE <= WARP_SIZE);                          \
    if (real.getSize(0) % (2 * FFTS_PER_WARP) == 0) {                   \
      dim3 blocks(ceil(ceil(real.getSize(0), 2 * FFTS_PER_WARP),        \
                       BATCH_UNROLL));                                  \
      dim3 threads(real.getSize(1) * FFTS_PER_WARP, 1, BATCH_UNROLL);   \
      detail::decimateInFrequency1DKernel<FFT_SIZE, 1, FFTS_PER_WARP, false> \
        <<<blocks, threads, 0, s>>>(real, complex);                     \
    } else {                                                            \
      dim3 blocks(ceil(real.getSize(0), 2 * BATCH_UNROLL));             \
      dim3 threads(real.getSize(1), 1, BATCH_UNROLL);                   \
      detail::decimateInFrequency1DKernel<FFT_SIZE, 1, 1, false>        \
        <<<blocks, threads, 0, s>>>(                                    \
          real, complex);                                               \
    }                                                                   \
    return FBFFTParameters::Success;                                    \
  }

#define SELECT_FBFFT_1D_DIF_GT_WARP_SIZE(FFT_SIZE, BATCH_UNROLL)        \
  if (real.getSize(1) == FFT_SIZE) {                                    \
    cuda_static_assert(FFT_SIZE > WARP_SIZE);                           \
    dim3 blocks(ceil(real.getSize(0), 2 * BATCH_UNROLL));               \
    dim3 threads(WARP_SIZE, 1, BATCH_UNROLL);                           \
    detail::decimateInFrequency1DKernel<FFT_SIZE, BATCH_UNROLL, 1, false> \
      <<<blocks, threads, 0, s>>>(real, complex);                       \
    return FBFFTParameters::Success;                                    \
  }

  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE( 2, 32, 16);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE( 4,  8,  8);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE( 8,  4,  4);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE(16,  4,  2);
  SELECT_FBFFT_1D_DIF_LE_WARP_SIZE(32,  4,  1);
  SELECT_FBFFT_1D_DIF_GT_WARP_SIZE(64,  4);
  SELECT_FBFFT_1D_DIF_GT_WARP_SIZE(128, 4);
  SELECT_FBFFT_1D_DIF_GT_WARP_SIZE(256, 2);

#undef SELECT_FBFFT_1D_DIF_LE_WARP_SIZE
#undef SELECT_FBFFT_1D_DIF_GT_WARP_SIZE

  return FBFFTParameters::UnsupportedSize;
}

} } } // namespace
