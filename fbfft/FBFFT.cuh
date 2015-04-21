// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "cuda/Complex.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/fbfft/FBFFTParameters.h"

namespace facebook { namespace cuda { namespace fbfft {

template <int BatchDims>
FBFFTParameters::ErrorCode fbfft1D(
  cuda::DeviceTensor<float, BatchDims + 1>& real,
  cuda::DeviceTensor<float, BatchDims + 2>& complex,
  const int padL = 0,
  cudaStream_t s = 0);

template <int BatchDims>
FBFFTParameters::ErrorCode fbifft1D(
  cuda::DeviceTensor<float, BatchDims + 1>& real,
  cuda::DeviceTensor<float, BatchDims + 2>& complex,
  const int padL = 0,
  cudaStream_t s = 0);

// Padding: only pad left and up since right and bottom padding is
// always done explicitly to fill up to a basis of size 2^k
// It is the responsibility of the caller to call an FFT of size 2^(k+1) if
// the 2^k is not sufficient to accomodate both the desired size and padding
// up/down/left/right.
template <int BatchDims>
FBFFTParameters::ErrorCode fbfft2D(
  cuda::DeviceTensor<float, BatchDims + 2>& real,
  cuda::DeviceTensor<float, BatchDims + 3>& complex,
  const int padL = 0,
  const int padU = 0,
  cudaStream_t s = 0);

// Padding only occurs on the real input
template <int BatchDims>
FBFFTParameters::ErrorCode fbfft2D(
  cuda::DeviceTensor<cuda::Complex, BatchDims + 2>& src,
  cuda::DeviceTensor<cuda::Complex, BatchDims + 2>& dst,
  cudaStream_t s = 0);

template <int BatchDims>
FBFFTParameters::ErrorCode fbifft2D(
  cuda::DeviceTensor<float, BatchDims + 3>& src,
  cuda::DeviceTensor<float, BatchDims + 3>& dst,
  cudaStream_t s = 0);
template <int BatchDims>
FBFFTParameters::ErrorCode fbifft2D(
  cuda::DeviceTensor<cuda::Complex, BatchDims + 2>& complexSrc,
  cuda::DeviceTensor<float, BatchDims + 2>& realDst,
  const int padL = 0,
  const int padU = 0,
  cudaStream_t s = 0);

} } } // namespace
