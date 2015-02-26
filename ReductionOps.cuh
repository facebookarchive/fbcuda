// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <cuda.h>

/** @file
    A collection of commutative operations for warp-wide and
    block-wide reduction functions.
    While for floating point math some of these operations are not
    commutative, when used in reduction, they will be applied in a
    deterministic manner.
    One can imagine any other function that follows CRDT-type rules
    could apply here.
*/

namespace facebook { namespace cuda {

template <typename T>
struct Sum {
  __host__ __device__ __forceinline__ T operator()(T a, T b) {
    return a + b;
  }
};

template <typename T>
struct Max {
  __host__ __device__ __forceinline__ T operator()(T a, T b) {
    return max(a, b);
  }
};

template <typename T>
struct Min {
  __host__ __device__ __forceinline__ T operator()(T a, T b) {
    return min(a, b);
  }
};

} } // namespace
