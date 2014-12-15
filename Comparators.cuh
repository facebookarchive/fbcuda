// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <cuda.h>

namespace facebook { namespace cuda {

/**
   Prototype:

   ~~~{.cpp}
   template <typename T>
   struct Comparator {
     static __device__ __forceinline__ bool compare(const T lhs, const T rhs);
   };
   ~~~
*/
template <typename T>
struct GreaterThan {
  static __device__ __forceinline__ bool compare(const T lhs, const T rhs) {
    return (lhs > rhs);
  }
};

template <typename T>
struct LessThan {
  static __device__ __forceinline__ bool compare(const T lhs, const T rhs) {
    return (lhs < rhs);
  }
};

} } // namespace
