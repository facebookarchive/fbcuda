// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/Comparators.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/ShuffleTypes.cuh"

namespace facebook { namespace cuda {

namespace detail {

template <typename T, typename Comparator>
__device__ __forceinline__ T shflSwap(const T x, int mask, int dir) {
  T y = shfl_xor(x, mask);
  return Comparator::compare(x, y) == dir ? y : x;
}

} // namespace

/// Defines a bitonic sort network to exchange 'V' according to
/// `SWAP()`'s compare and exchange mechanism across the warp, ordered
/// according to the comparator `comp`. In other words, if `comp` is
/// `GreaterThan<T>`, then lane 0 will contain the highest `val`
/// presented across the warp
///
/// See also 
/// http://on-demand.gputechconf.com/gtc/2013/presentations/S3174-Kepler-Shuffle-Tips-Tricks.pdf
template <typename T, typename Comparator>
__device__ T warpBitonicSort(T val) {
  const int laneId = getLaneId();
  // 2
  val = detail::shflSwap<T, Comparator>(
    val, 0x01, getBit(laneId, 1) ^ getBit(laneId, 0));

  // 4
  val = detail::shflSwap<T, Comparator>(
    val, 0x02, getBit(laneId, 2) ^ getBit(laneId, 1));
  val = detail::shflSwap<T, Comparator>(
    val, 0x01, getBit(laneId, 2) ^ getBit(laneId, 0));

  // 8
  val = detail::shflSwap<T, Comparator>(
    val, 0x04, getBit(laneId, 3) ^ getBit(laneId, 2));
  val = detail::shflSwap<T, Comparator>(
    val, 0x02, getBit(laneId, 3) ^ getBit(laneId, 1));
  val = detail::shflSwap<T, Comparator>(
    val, 0x01, getBit(laneId, 3) ^ getBit(laneId, 0));

  // 16
  val = detail::shflSwap<T, Comparator>(
    val, 0x08, getBit(laneId, 4) ^ getBit(laneId, 3));
  val = detail::shflSwap<T, Comparator>(
    val, 0x04, getBit(laneId, 4) ^ getBit(laneId, 2));
  val = detail::shflSwap<T, Comparator>(
    val, 0x02, getBit(laneId, 4) ^ getBit(laneId, 1));
  val = detail::shflSwap<T, Comparator>(
    val, 0x01, getBit(laneId, 4) ^ getBit(laneId, 0));

  // 32
  val = detail::shflSwap<T, Comparator>(
    val, 0x10, getBit(laneId, 4));
  val = detail::shflSwap<T, Comparator>(
    val, 0x08, getBit(laneId, 3));
  val = detail::shflSwap<T, Comparator>(
    val, 0x04, getBit(laneId, 2));
  val = detail::shflSwap<T, Comparator>(
    val, 0x02, getBit(laneId, 1));
  val = detail::shflSwap<T, Comparator>(
    val, 0x01, getBit(laneId, 0));

  return val;
}

} } // namespace
