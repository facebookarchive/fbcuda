// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/ComputeCapabilities.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/ReductionOps.cuh"
#include "cuda/WarpBitonicSort.cuh"

#include <cuda.h>

/** @file
    Utilities for reducing a value with respect to a commutative
    operation across a warp, using warp shuffles.
    While floating-point reduction operators are not strictly
    commutative, the reduction is always done in a deterministic
    manner. The only property that we expect from the operator in a
    mathematical sense is that it is commutative.
*/

namespace facebook { namespace cuda {

/// Reduce a value across a warp by applying the commutative function
/// `Op`. All threads in the warp receive the reduced value.
/// Assumes that all threads in the warp are participating in the
/// reduction.
template <typename T, typename Op>
__device__ __forceinline__ T warpReduce(T val, Op op) {
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    val = op(val, __shfl_xor(val, mask));
  }

  return val;
}

/// Sums a register value across all warp threads
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  return warpReduce<T, Sum<T> >(val, Sum<T>());
}

/// Finds the maximum `val` across the warp
template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
  return warpReduce<T, Max<T> >(val, Max<T>());
}

/// Finds the minimum `val` across the warp
template <typename T>
__device__ __forceinline__ T warpReduceMin(T val) {
  return warpReduce<T, Max<T> >(val, Min<T>());
}

/// Determine if two warp threads have the same value (a collision).
template <typename T>
__device__ __forceinline__ bool warpHasCollision(T val) {
  // -sort all values
  // -compare our lower neighbor's value against ourselves (excepting
  //  the first lane)
  // -if any lane as a difference of 0, there is a duplicate
  //  (excepting the first lane)
  val = warpBitonicSort<T, LessThan<T> >(val);
  const T lower = __shfl_up(val, 1);

  // Shuffle for lane 0 will present its same value, so only
  // subsequent lanes will detect duplicates
  const bool dup = (lower == val) && (getLaneId() != 0);
  return (__any(dup) != 0);
}

/// Determine if two warp threads have the same value (a collision),
/// and returns a bitmask of the lanes that are known to collide with
/// other lanes. Not all lanes that are mutually colliding return a
/// bit; all lanes with a `1` bit are guaranteed to collide with a
/// lane with a `0` bit, so the mask can be used to serialize
/// execution for lanes that collide with others.
/// (mask | (mask >> 1)) will yield all mutually colliding lanes.
template <typename T>
__device__ __forceinline__ unsigned int warpCollisionMask(T val) {
  // -sort all (lane, value) pairs on value
  // -compare our lower neighbor's value against ourselves (excepting
  //  the first lane)
  // -if any lane as a difference of 0, there is a duplicate
  //  (excepting the first lane)
  // -shuffle sort (originating lane, dup) pairs back to the original
  //  lane and report
  Pair<T, int> pVal(val, getLaneId());
  pVal = warpBitonicSort<Pair<T, int>, LessThan<Pair<T, int> > >(pVal);

  // If our neighbor is the same as us, we know our thread's value is
  // duplicated. All except for lane 0, since shfl will present its
  // own value (and if lane 0's value is duplicated, lane 1 will pick
  // that up)
  const unsigned long lower = __shfl_up(pVal.k, 1);
  Pair<int, bool> dup(pVal.v,
                      (lower == pVal.k) && (getLaneId() != 0));

  // Sort back based on lane ID so each thread originally knows
  // whether or not it duplicated
  dup = warpBitonicSort<Pair<int, bool>,
                        LessThan<Pair<int, bool> > >(dup);
  return __ballot(dup.v);
}

} } // namespace
