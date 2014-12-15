// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <math_constants.h>

namespace facebook { namespace cuda {

/// Numeric limits for CUDA
template <typename T>
struct NumericLimits {};

template<>
struct NumericLimits<float> {
  /// The minimum possible valid float (i.e., not NaN)
  __device__ __forceinline__ static float minPossible() {
    return -CUDART_INF_F;
  }

  /// The maximum possible valid float (i.e., not NaN)
  __device__ __forceinline__ static float maxPossible() {
    return CUDART_INF_F;
  }
};

template<>
struct NumericLimits<int> {
  /// The minimum possible int
  __device__ __forceinline__ static int minPossible() {
    return INT_MIN;
  }

  /// The maximum possible int
  __device__ __forceinline__ static int maxPossible() {
    return INT_MAX;
  }
};

template<>
struct NumericLimits<unsigned int> {
  /// The minimum possible unsigned int
  __device__ __forceinline__ static unsigned int minPossible() {
    return 0;
  }

  /// The maximum possible unsigned int
  __device__ __forceinline__ static unsigned int maxPossible() {
    return UINT_MAX;
  }
};

} } // namespace
