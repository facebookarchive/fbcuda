// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/ComputeCapabilities.cuh"
#include "cuda/NumericLimits.cuh"
#include "cuda/RegisterUtils.cuh"
#include "cuda/SmallSort.cuh"
#include "cuda/TopK.cuh"

#include <assert.h>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <cuda.h>
#include <device_functions.h>
#include <math_constants.h>
#include <stdio.h>

/** @file
    
    CUDA device code routines for finding all top K float elements in
    descending order in a set using the above top-Kth radix selection
    plus warp coherent bitonic sorting.
*/
namespace facebook { namespace cuda {

namespace detail {

/// Returns the index into the array that this lane will write. If this
/// lane is not responsible for writing a value, this will return -1.
__device__ __forceinline__ int
laneWillWrite(float val, float topK, int& topKToWrite, int& next) {
  // Do we have a < top K value? Those must be written.
  // If we have a == top K value, only some of these may be written.
  const bool weHaveLessThanTopK = (val > topK);
  const bool weHaveEqualToTopK = (val == topK);

  // All threads with an on bit in this will write out to `out`
  const unsigned warpHasLessThanTopK = __ballot(weHaveLessThanTopK);

  // Only the first topKToWrite threads with an on bit in this will
  // write out to `out`
  unsigned warpHasEqualToTopK = __ballot(weHaveEqualToTopK);

  // We have to figure out which ones are the first topKToWrite ones
  // though
  const bool weWillWriteEqualToTopK =
    (__popc(getLaneMaskLt() & warpHasEqualToTopK) < topKToWrite) &&
    weHaveEqualToTopK;

  // Tell all threads which ones will write out == top K elements
  warpHasEqualToTopK = __ballot(weWillWriteEqualToTopK);

  // Update the number of actual == top K elements to find remaining
  topKToWrite -= __popc(warpHasEqualToTopK);
  assert(topKToWrite >= 0);

  // Only the lanes with bits set in this mask will write out elements
  const unsigned warpWillWrite = warpHasLessThanTopK | warpHasEqualToTopK;

  // How many threads are writing before us? This will define our
  // output order.
  const unsigned numLanesBeforeUs = __popc(getLaneMaskLt() & warpWillWrite);

  // Thus, next + numLanesBeforeUs is the index into which we'll
  // write our value, if this lane wants to write a value.
  const int writeIndex = next + numLanesBeforeUs;

  // Advance where the next values go by how many values the current
  // warp wrote out
  next += __popc(warpWillWrite);

  // Only if this lane bit is on will we write something out
  return getBit(warpWillWrite, getLaneId()) ? writeIndex : -1;
}

/// For a given warp, find and write out the top-k highest floating
/// point values in [start, end) to [out, out + k). The list written
/// out occurs in the original source order (by original
/// index). Returns the k-th highest element seen.
/// Handles all floats except NaNs.
/// Implementation for large arrays such that there are more elements
/// than warp threads.
__device__ float
warpFindTopKElementsIndexOrder(const DeviceTensor<float, 1>& data,
                               DeviceTensor<float, 1>& out,
                               int k) {
  // First, have all warp threads find the top Kth element.
  const Pair<float, int> topKthElement = warpFindTopKthElement(data, k);

  // The next offset to write into `out`
  int next = 0;

  // Number of remaining == topKthElement values the warp still has to
  // write (because there can be duplicates)
  int topKToWrite = topKthElement.v;

  for (int index = getLaneId(); index < data.getSize(0); index += WARP_SIZE) {
    const float val = data[index];
    const int idx = laneWillWrite(val, topKthElement.k, topKToWrite, next);

    // Does this lane have a value to write?
    if (idx != -1) {
      out[idx] = val;
    }
  }

  // We should have written out all the == top K elements. However,
  // only threads that were within bounds will have the proper values
  // of these, so share from the thread within the first lane, which
  // is guaranteed to participate in all array loops
  assert(__shfl(topKToWrite, 0) == 0);
  assert(__shfl(next, 0) == k);

  return topKthElement.k;
}

/// Version of warpFindTopKElementsUnorderedLarge, except also writes
/// out the K indices chosen from `data` into `indices`.
template <typename IndexType>
__device__ float
warpFindTopKElementsIndexOrder(const DeviceTensor<float, 1>& data,
                               DeviceTensor<float, 1>& out,
                               DeviceTensor<IndexType, 1>& indices,
                               int k) {
  // First, have all warp threads find the top Kth element.
  const Pair<float, int> topKthElement = warpFindTopKthElement(data, k);

  // The next offset to write into `out`
  int next = 0;

  // Number of remaining == topKthElement values the warp still has to
  // write (because there can be duplicates)
  int topKToWrite = topKthElement.v;

  for (int index = getLaneId(); index < data.getSize(0); index += WARP_SIZE) {
    const float val = data[index];
    const int idx = laneWillWrite(val, topKthElement.k, topKToWrite, next);

    // Does this lane have a value to write?
    if (idx != -1) {
      out[idx] = val;
      indices[idx] = (IndexType) index;
    }
  }

  // We should have written out all the == top K elements. However,
  // only threads that were within bounds will have the proper values
  // of these, so share from the thread within the first lane, which
  // is guaranteed to participate in all array loops
  assert(__shfl(topKToWrite, 0) == 0);
  assert(__shfl(next, 0) == k);

  return topKthElement.k;
}

/// For a given warp, find and write out the top-k highest floating
/// point values in [start, end) to [out, out + k). The list written
/// out is ordered.
/// Handles all floats except NaNs.
__device__ void
warpFindTopKElementsValueOrderSmall(const DeviceTensor<float, 1>& data,
                                    DeviceTensor<float, 1>& out,
                                    int k) {
  // We only handle in-warp sorting up to a max size; above this size,
  // the radix selection strategy wins.
  assert(data.getSize(0) <= 3 * WARP_SIZE);
  // There should be enough values to return the k-th highest.
  assert(k > 0 && k <= data.getSize(0));

  const int lane = getLaneId();

#define HANDLE_SIZE(N)                                          \
  if (data.getSize(0) <= N * WARP_SIZE) {                       \
    float val[N];                                               \
    WarpRegisterLoaderUtils<float, N>::load(                    \
      val, data, NumericLimits<float>::minPossible());          \
                                                                \
    float sorted[N];                                                    \
    warpSortRegisters<float, GreaterThan<float>, N>(val, sorted);       \
                                                                        \
    WarpRegisterLoaderUtils<float, N>::save(out, sorted, k);            \
  }

  HANDLE_SIZE(1);
  HANDLE_SIZE(2);
  HANDLE_SIZE(3);

#undef HANDLE_SIZE
}

/// Version of warpFindTopKElementsOrderedSmall that also writes out
/// the indices in `data` of the K elements chosen into `indices`.
template <typename IndexType>
__device__ void
warpFindTopKElementsValueOrderSmall(const DeviceTensor<float, 1>& data,
                                    DeviceTensor<float, 1>& out,
                                    DeviceTensor<IndexType, 1>& indices,
                                    int k) {
  // We only handle in-warp sorting up to a max size; above this size,
  // the radix selection strategy wins.
  assert(data.getSize(0) <= 3 * WARP_SIZE);
  // There should be enough values to return the k-th highest.
  assert(k > 0 && k <= data.getSize(0));

  const int lane = getLaneId();

#define HANDLE_SIZE(N)                                          \
  if (data.getSize(0) <= N * WARP_SIZE) {                       \
    Pair<float, IndexType> val[N];                              \
    WarpRegisterPairLoaderUtils<float, IndexType, N>::load(     \
      val, data,                                                \
      NumericLimits<float>::minPossible(),                      \
      NumericLimits<IndexType>::minPossible());                 \
                                                                \
    Pair<float, IndexType> sorted[N];                           \
    warpSortRegisters<Pair<float, IndexType>,                   \
                      GreaterThan<Pair<float, IndexType> >,     \
                      N>(val, sorted);                          \
                                                                \
    WarpRegisterPairLoaderUtils<float, IndexType, N>::save(     \
      out, indices, sorted, k);                                 \
  }

  HANDLE_SIZE(1);
  HANDLE_SIZE(2);
  HANDLE_SIZE(3);

#undef HANDLE_SIZE
}

/// For a given warp, find and write out the top-k highest floating
/// point values in [start, end) to [out, out + k). The list written
/// out is ordered.
/// Handles all floats except NaNs.
/// Implementation for large arrays such that there are more elements
/// than warp threads.
__device__ void
warpFindTopKElementsValueOrderLarge(const DeviceTensor<float, 1>& data,
                                    DeviceTensor<float, 1>& out,
                                    int k) {
  // We only have a sorting implementation that works up to k <= 4 *
  // warpSize.
  assert(k <= 4 * WARP_SIZE);

  // Find and write out the elements in index order
  warpFindTopKElementsIndexOrder(data, out, k);

  // Sort the elements in [out, out + k) based on float order
  bool sorted = warpSort<float, GreaterThan<float> >(out, out);
  assert(sorted);
}

/// Version of warpFindTopKElementsOrderedLage that also writes out the
/// indices in `data` of the K elements chosen into `indices`.
template <typename IndexType>
__device__ void
warpFindTopKElementsValueOrderLarge(const DeviceTensor<float, 1>& data,
                                    DeviceTensor<float, 1>& out,
                                    DeviceTensor<IndexType, 1>& indices,
                                    int k) {
  // We only have a sorting implementation that works up to k <= 4 *
  // warpSize.
  assert(k <= 4 * WARP_SIZE);

  // Find and write out the elements in potentially unsorted order
  detail::warpFindTopKElementsIndexOrder<IndexType>(data, out, indices, k);

  // Sort the elements in [out, out + k) / [indices, indices + k) as
  // keys/values
  bool sorted =
    warpSort<float, IndexType, GreaterThan<Pair<float, IndexType> > >(
      out, indices, out, indices);
  assert(sorted);
}

} // detail

/// For a given warp, find and write out the top-k highest floating
/// point values in [start, end) to [out, out + k). The list written
/// out is ordered based on original index order. Handles all floats
/// except NaNs.
__device__ void
warpFindTopKElementsIndexOrder(const DeviceTensor<float, 1>& data,
                               DeviceTensor<float, 1>& out,
                               int k) {
  assert(out.getSize(0) >= k);
  detail::warpFindTopKElementsIndexOrder(data, out, k);
}

/// Version of warpFindTopKElementsOrdered which also writes out the
/// indices of the found top elements from `data`. The list written out
/// is ordered based on original index order. Handles all floats except
/// NaNs.
/// Supports writing out float or integer indices.
template <typename IndexType>
__device__ void
warpFindTopKElementsIndexOrder(const DeviceTensor<float, 1>& data,
                               DeviceTensor<float, 1>& out,
                               DeviceTensor<IndexType, 1>& indices,
                               int k) {
  assert(out.getSize(0) >= k && indices.getSize(0) >= k);

  detail::warpFindTopKElementsIndexOrder<IndexType>(
    data, out, indices, k);
}

/// For a given warp, find and write out the top-k highest floating
/// point values in [start, end) to [out, out + k). The list written
/// out is ordered based on float value. Handles all floats except
/// NaNs.
__device__ void
warpFindTopKElementsValueOrder(const DeviceTensor<float, 1>& data,
                               DeviceTensor<float, 1>& out,
                               int k) {
  assert(out.getSize(0) >= k);
  assert(k <= 4 * WARP_SIZE); // Max size handled at the moment

  // In-register warp sorting is faster up to 3 x warpSize input
  if (data.getSize(0) <= 3 * WARP_SIZE) {
    detail::warpFindTopKElementsValueOrderSmall(data, out, k);
  } else {
    detail::warpFindTopKElementsValueOrderLarge(data, out, k);
  }
}

/// Version of warpFindTopKElementsOrdered which also writes out the
/// indices of the found top elements from `data`. The list written out
/// is ordered based on float value. Handles all floats except NaNs.
/// Supports writing out float or integer indices.
template <typename IndexType>
__device__ void
warpFindTopKElementsValueOrder(const DeviceTensor<float, 1>& data,
                               DeviceTensor<float, 1>& out,
                               DeviceTensor<IndexType, 1>& indices,
                               int k) {
  assert(out.getSize(0) >= k && indices.getSize(0) >= k);
  assert(k <= 4 * WARP_SIZE); // Max size handled at the moment

  // In-register warp sorting is faster up to 3 x warpSize input
  if (data.getSize(0) <= 3 * WARP_SIZE) {
    detail::warpFindTopKElementsValueOrderSmall<IndexType>(
      data, out, indices, k);
  } else {
    detail::warpFindTopKElementsValueOrderLarge<IndexType>(
      data, out, indices, k);
  }
}

} } // namespace
