// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "cuda/Complex.cuh"
#include "cuda/DeviceTensor.cuh"

namespace facebook { namespace cuda { namespace fbfft {

class FBFFTParameters {
 public:
  enum ErrorCode {
    Success = 0,
    UnsupportedSize = 1,
    UnsupportedDimension = 2
  };

  FBFFTParameters& forward() {
    direction_ = true;
    return *this;
  }

  FBFFTParameters& inverse() {
    direction_ = false;
    return *this;
  }

  FBFFTParameters& normalize(bool n) {
    normalize_ = n;
    return *this;
  }

  bool forwardFFT() const { return  direction_; }
  bool inverseFFT() const { return !direction_; }
  bool normalizeFFT() const { return normalize_; }

 private:
  bool direction_;
  bool normalize_;
};

template <int BatchDims>
FBFFTParameters::ErrorCode fbfft1D(
  cuda::DeviceTensor<float, BatchDims + 1>& real,
  cuda::DeviceTensor<float, BatchDims + 2>& complex,
  cudaStream_t s = 0);

template <int BatchDims>
FBFFTParameters::ErrorCode fbifft1D(
  cuda::DeviceTensor<float, BatchDims + 1>& real,
  cuda::DeviceTensor<float, BatchDims + 2>& complex,
  cudaStream_t s = 0);

template <int BatchDims>
FBFFTParameters::ErrorCode fbfft2D(
  cuda::DeviceTensor<float, BatchDims + 2>& real,
  cuda::DeviceTensor<float, BatchDims + 3>& complex,
  cudaStream_t s = 0);

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
  cudaStream_t s = 0);

} } } // namespace
