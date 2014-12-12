// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/ComputeCapabilities.cuh"
#include "cuda/DeviceTensor.cuh"

#include <cuda.h>

namespace facebook { namespace cuda {

// Sums a register value across all warp threads
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor(val, mask);
  }

  return val;
}

// Finds the maximum `val` across the warp
template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    val = max(__shfl_xor(val, mask), val);
  }

  return val;
}

// Finds the minimum `val` across the warp
template <typename T>
__device__ __forceinline__ T warpReduceMin(T val) {
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    val = min(__shfl_xor(val, mask), val);
  }

  return val;
}

// Bitwise or reduction across the warp
template <typename T>
__device__ __forceinline__ T warpReduceBitwiseOr(T val) {
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    val = __shfl_xor(val, mask) | val;
  }

  return val;
}

} } // namespace
