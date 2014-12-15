// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <cuda.h>

namespace facebook { namespace cuda {

/// A simple pair type for CUDA device usage
template <typename K, typename V>
struct Pair {
  __host__ __device__ __forceinline__ Pair() {
  }

  __host__ __device__ __forceinline__ Pair(K key, V value)
      : k(key), v(value) {
  }

  __host__ __device__ __forceinline__ bool
  operator==(const Pair<K, V>& rhs) const {
    return (k == rhs.k) && (v == rhs.v);
  }

  __host__ __device__ __forceinline__ bool
  operator!=(const Pair<K, V>& rhs) const {
    return !operator==(rhs);
  }

  __host__ __device__ __forceinline__ bool
  operator<(const Pair<K, V>& rhs) const {
    return (k < rhs.k) || ((k == rhs.k) && (v < rhs.v));
  }

  __host__ __device__ __forceinline__ bool
  operator>(const Pair<K, V>& rhs) const {
    return (k > rhs.k) || ((k == rhs.k) && (v > rhs.v));
  }

  K k;
  V v;
};

} } // namespace
