// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/ComputeCapabilities.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/WarpBitonicSort.cuh"
#include "cuda/WarpReductions.cuh"

#include <assert.h>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <cuda.h>
#include <device_functions.h>
#include <math_constants.h>
#include <stdio.h>

/** @file
   
    CUDA device code routines for finding the top-Kth float element
    in a set in O(N) time using radix selection. Uses no scratch space
    and does not modify inputs.

    Right now only contains versions to do this work in a kernel
    utilizing only the threads in a warp in a almost entirely
    warp-coherent manner. An entire warp must be given.
*/

namespace facebook { namespace cuda {

namespace detail {

/// Initialize an array to a value
template <int N, typename T>
__device__ __forceinline__ void setArray(T arr[N], T val) {
  for (int i = 0; i < N; ++i) {
    arr[i] = val;
  }
}

/// In order to force register usage of the bucket count array, we have to
/// unroll the increment selection. Otherwise, local memory is used for
/// counts[] which severely degrades performance.
__device__ __forceinline__ void incrementArray(int val, int counts[16]) {
#define BUCKET_CASE(UNUSED1, I, UNUSED2)        \
  case I:                                       \
    counts[I]++;                                \
    break;

  switch (val) {
    BOOST_PP_REPEAT(16, BUCKET_CASE, 0);
    default:
      break;
  }

#undef BUCKET_CASE
}

/**
   
   We use a most significant to least significant radix selection on
   the float values, which requires at most sizeof(float) * 2 scans
   through the array, one for each nybble.

   In order to use radix selection, we use the property that for
   positive floating-point values f1 and f2:

   ~~~
   f1 > f2 <=> *(int*)&f1 > *(int*)&f2.
   ~~~
   
   Something similar is true for negative floating point values f1
   and f2 after zero-ing the leading sign bit, and except that the
   order is reversed:

   ~~~
   f1 > f2 <=> (*(int*)f1 & 0x7fffffff) < (*(int*)f2 & 0x7fffffff).
   ~~~
   
   This is true even for +/-inf and for denormalized floats.
   Negative zero is a special case. Selection by radix will give us
   that -0.0f < +0.0f, which is not true for IEEE fp comparison. We
   handle this special case when we return the answer seen, not in
   comparing values here.
   
   +NaNs will lead all positive values, and -NaNs will be minimal
   values (non-canonical NaNs, if they exist, will be sorted
   according to this).

   The focal point of the radix selection algorithm is the use of
   countNybbles and the CHECK_NYBBLE macro.

   The idea is that we starting out, we don't know where the Kth
   highest element lives, so we have to consider *every* float in
   the input. We look at the most significant nybble, and each
   thread counts into 16 buckets the number of floats in its subset
   of data with that leading nybble. This is done by countNybbles.
   countNybble takes as arguments `desired` and `desiredMask`. It only
   looks at values `v` such that (v & desiredMask) == desired. By
   default, both are 0, so it will look at every float. nybbleCheckPos
   is the current nybble that is beinig bucketed. It starts at 28,
   meaning we're first looking at the most significant nybble.

   countNybbles will add a count of nybble distribution to 16 buckets.
   One iteration through, by counting the distribution of the leading
   nybble in each float, we figure out what leading nybble the Kth
   highest float must have. As an example, let's say that K is 10. By
   counting the distribution of leading nybbles in the inputs, say we
   get:

   ~~~
   0x2: 1 0x5: 2 0x6: 2 0x7: 11
   ~~~
   
   In this case, none of the floats are negative (otherwise, they
   would have leading nybble 0x8 -> 0xf). Since we're looking for the
   10th highest float, that cannot have leading nybble 0x2, 0x5 or 0x6
   since those counts are less than 10. We walk through the buckets in
   order, and we warp reduce the counts across all threads to one
   count when it comes time to look in a particular bucket.
   
   Thus, the 10th highest float must have leading nybble 0x7. The
   problem then becomes, for the next iteration, finding the (10 -
   (1+2+2)) = 5th highest float with leading nybble 0x7. Since the
   count for the nybble 0x7 is not 1, we don't know the actual answer
   yet, and we have to continue.

   Next iteration through, we no longer have to count every float,
   just those with leading nybble 0x7 (i.e., floats interpreted as a
   bit pattern v such that (v & desiredMask) == desired. Even though
   we have to physically scan the entire input, we are only counting a
   subset of it.

   So:
   - desired changes from 0 -> 0x70000000, and
   - desiredMask changes from 0 -> 0xf0000000.

   We continue, and count up the floats with leading nybble 0x7,
   getting counts:

   ~~~
   0x(7)1: 1 0x(7)9: 2 0x(7)b: 1 0x(7)c: 1 0x(7)f: 6
   ~~~

   This means that the count of all floats with the prefix 0x7fyyyyyy
   is 6. We're only bucketing counts by the second nybble now.
   
   Scanning through, the 5th highest float with prefix 0x7yyyyyyy must
   have prefix 0x7cyyyyyy, since from lowest to highest above, we
   reach 5 in bucket 0xc.
   
   Thus, the 10th highest float in the entire set is the unique float
   with prefix 0x7cyyyyyy. This is unique because the count for this
   bucket is 1.
   
   If we get through all nybbles to the least significant nybble and
   still have a count > 1, then that means that the Kth highest
   element is not unique. For example, in the set 2 2 3 3 3, the 2nd
   highest element is 3, which is duplicated 3 times.

   Since the MSN contains the sign bit, we have to first look at
   buckets 0-7 to see if the Kth highest float is positive. If so,
   then we continue looking only at positive floats. If not, then we
   continue looking only at negative floats, but in reverse order.

   Eventually we find a unique Kth highest element if the count is 1
   in our bucket, or we end at the LSN with a duplicate count, in
   which case the Kth highest element is not unique.

   Performs a histogram count of the nybbles that occur at the bit
   position `nybbleCheckPos`, but only for those ints that match (x &
   `desiredMask`) == `desired`.
   
   In other words, if bits [31, `nybbleCheckPos` + 4] match those in
   `desired`, then return the contents of bits [`nybbleCheckPos` + 3,
   `nybbleCheckPos`].
*/
template <typename T, int N, int ILP>
__device__ __forceinline__ void
countNybbles(int counts[N],
             unsigned desired,
             unsigned desiredMask,
             int nybbleCheckPos,
             const DeviceTensor<T, 1>& data) {
  // Clear out counts from a previous round
  setArray<N>(counts, 0);

  // Treat floats as unsigned ints, since we're counting raw nybble
  // values
  unsigned vals[ILP];
  setArray<N>(vals, 0U);

  // Handle ILP portion
  int index = getLaneId();

  // Distribute index loop among threads and unroll by ILP, each thread
  // operates on interleaved indices getLaneId() + i * WARP_SIZE.
  if (ILP > 1) {
    for ( ;
          index + (ILP - 1) * WARP_SIZE < data.getSize(0);
          index += WARP_SIZE * ILP) {
      for (int i = 0; i < ILP; ++i) {
        vals[i] = data[index + i * WARP_SIZE].template ldgAs<unsigned>();
      }

      for (int i = 0; i < ILP; ++i) {
        const unsigned val = vals[i];

        // We only consider values that match the bits we're looking
        // for in `desired`, since we've already ruled out other values
        if ((val & desiredMask) == desired) {
          // Add to our count of nybbles seen
          const unsigned nybble = getBitfield(val, nybbleCheckPos, 4);

          // Add to our counts (unrolled to force use of registers for
          // `counts`.
          incrementArray(nybble, counts);
        }
      }
    }
  }

  // Handle remainder
  for ( ; index < data.getSize(0); index += WARP_SIZE) {
    const unsigned val = data[index].template ldgAs<unsigned>();

    // We only consider values that match the bits we're looking
    // for in `desired`, since we've already ruled out other values
    if ((val & desiredMask) == desired) {
      // Add to our count of nybbles seen
      const unsigned nybble = getBitfield(val, nybbleCheckPos, 4);

      // Add to our counts (unrolled to force use of registers for
      // `counts`.
      incrementArray(nybble, counts);
    }
  }
}

/**
   A warp coherent implementation that finds a value in the data such
   that the floats, treated as uints 'v' match the bit pattern such
   that (v & desiredMask) == desired.

   If the answer found is -0.0f, because -0.0f == +0.0f, it is
   possible that there are multiple +0.0f results that we've ignored
   in comparing by radix, since radix-wise +0.0f > -0.0f which is not
   true for IEEE fp.

   Thus, if the answer found is -0.0f, then we have to include the
   count of all +0.0fs present in the duplicate count, in order to
   treat the comparison the same way that normal sorting mechanisms
   will treat it.

   If the found result is not -0.0f, returns the value found and
   `dupCount` as the pair's value.

   If the found result is -0.0f, returns the value found and
   `dupCount` plus the number of +0.0f in the data as the pair's
   value.

   The value need not be unique, but the warp as a whole will return
   the highest value seen across the warp.
*/
__device__ __forceinline__ Pair<float, int>
findAnswer(const DeviceTensor<float, 1>& data,
           unsigned desired,
           unsigned desiredMask,
           int dupCount) {
  // Each thread will scan for values with the desired prefix, and
  // then we gather the value (if any) across all threads in the warp
  // using a max() reduction between the values and -inf. If there is
  // an answer, it should be greater than -inf (unless it is -inf),
  // and the one reduced result should be the solution.
  float found = -CUDART_INF_F;

  // TODO: ILP?
  for (int index = getLaneId(); index < data.getSize(0); index += WARP_SIZE) {
    float val = data[index].ldg();
    if (((unsigned) __float_as_int(val) & desiredMask) == desired) {
      found = val;
    }
  }

  const float max = warpReduceMax(found);

  if (__float_as_int(max) == __float_as_int(-0.0f)) {
    // Special case negative zero, in order to handle the +0.0f ==
    // -0.0f property
    int posZeroCount = 0;

    for (int index = getLaneId(); index < data.getSize(0); index += WARP_SIZE) {
      if (__float_as_int(data[index].ldg()) == __float_as_int(0.0f)) {
        ++posZeroCount;
      }
    }

    posZeroCount = warpReduceSum(posZeroCount);
    return Pair<float, int>(max, posZeroCount + dupCount);
  }

  return Pair<float, int>(max, dupCount);
}

/**
   Finds the Kth highest floating point value in a linear array [arr,
   end) without modifying the data and without temporary storage
   except for registers. K starts at 1. All threads in the warp will
   return the value. Handles all floats except NaNs.

   This function minimizes warp divergence.

   Implementation for small arrays such that the `(end - start) <=
   warpSize`.
*/
__device__ Pair<float, int>
warpFindTopKthElementSmall32(const DeviceTensor<float, 1>& data, int k) {
  // The array should fit within the warp size.
  assert(data.getSize(0) <= WARP_SIZE);
  // There should be enough values to return the k-th highest.
  assert(k > 0 && k <= data.getSize(0));

  const int lane = getLaneId();

  // For threads in the warp that have no element in the array, give
  // them -inf, so they'll sort to the end.
  float val = (lane < data.getSize(0)) ? data[lane] : -CUDART_INF_F;

  // Warp coherent sort! Handle negative zero, as does the radix code
  val = warpBitonicSort<float, GreaterThan<float> >(val);

  // Lane k - 1 now contains the kth highest element; broadcast it to
  // all threads in the warp
  const float topK = __shfl(val, k - 1);

  // Also return the number of lanes <= k - 1 that have this same topK
  // value; this is the number of duplicates present.
  const int numSeen =
    warpReduceSum((int) ((topK == val) && getLaneId() < k));

  return Pair<float, int>(topK, numSeen);
}

/**
   Finds the Kth highest floating point value in a linear array [arr,
   end) without modifying the data and without temporary storage
   except for registers.

   - K starts at 1.
   - All threads in the warp will return the value.
   - Handles all floats except NaNs.
   - Negative zero is specialized by findAnswer.
   - This function minimizes warp divergence.

   Implementation for large arrays such that there are more elements
   than warp threads.
*/
__device__ Pair<float, int>
warpFindTopKthElementLarge(const DeviceTensor<float, 1>& data, int k) {
  // There should be enough values to return the k-th highest.
  assert(k > 0 && k <= data.getSize(0));

  // kNybbles is the number of possible values of a nybble (2^4).
#define kNybbles 16 // TODO: C++11 constexpr
  int nybbleCounts[kNybbles];

  // We are currently evaluating the nybble in this position (e.g.,
  // the nybble we're scanning is the one in bits
  // [`nybbleCheckPos` + 3, `nybbleCheckPos`]. Initially, we look at
  // the most significant nybble (28).
  int nybbleCheckPos = 28;
  // We only consider elements x such that (x & desiredMask) == desired
  // Initially, we consider all elements of the array, so the above
  // statement is true regardless of input.
  unsigned desired = 0;
  unsigned desiredMask = 0;

  // Accumulate leading nybble counts
  // TODO: select ILP value before starting
  detail::countNybbles<float, kNybbles, 1>(
    nybbleCounts, desired, desiredMask, nybbleCheckPos, data);

  // We are looking for the top kToFind-th element when iterating over
  // nybbles; this count gets reduced by elimination when counting
  // successive nybbles
  int kToFind = k;

  // For each nybble we're evaluating, in whatever order we're
  // evaluating them, we have to do the same work
#define CHECK_NYBBLE()                                                  \
  {                                                                     \
    /* Only reduce the bucket sum if we need to, when we need to */     \
    const int count = warpReduceSum(nybbleCounts[i]);                   \
                                                                        \
    if (count == 1 && kToFind == 1) {                                   \
      /* There is a unique answer to the top-Kth element */             \
      /* The unique answer contains the desired nybble at */            \
      /* bit positions [`nybbleCheckPos1 + 3, `nybbleCheckPos`] */      \
      return detail::findAnswer(                                        \
        data,                                                           \
        setBitfield(desired, i, nybbleCheckPos, 4),                     \
        setBitfield(desiredMask, 0xf, nybbleCheckPos, 4),               \
        1);                                                             \
    }                                                                   \
                                                                        \
    if (count >= kToFind) {                                             \
      /* The top-Kth element must contain this nybble. */               \
      /* Add it to the prefix we're looking for, and continue on */     \
      /* the next nybble at `nybbleCheckPos` - 4  */                    \
      desired = setBitfield(desired, i, nybbleCheckPos, 4);             \
      desiredMask = setBitfield(desiredMask, 0xf, nybbleCheckPos, 4);   \
      nybbleCheckPos -= 4;                                              \
      break;                                                            \
    }                                                                   \
                                                                        \
    kToFind -= count;                                                   \
  }

  // Figure out what leading nybble the k-th largest float should
  // have, and by extension, whether it is positive or negative.
  // Scan the MSN first, positive floats only (the sign bit is the
  // leading bit).
  for (int i = (kNybbles / 2) - 1; i >= 0; --i) {
    CHECK_NYBBLE();
  }

  bool kthLargestIsPositive = true;

  if (desiredMask == 0) {
    // The k-th largest float is negative.
    kthLargestIsPositive = false;

    // What leading nybble does it have?
    for (int i = kNybbles / 2; i < kNybbles; ++i) {
      CHECK_NYBBLE();
    }
  }

  for ( ; ; ) {
    // Now, we only consider floats with (f & mask) == desired.
    // TODO: select ILP value before starting
    detail::countNybbles<float, kNybbles, 1>(
      nybbleCounts, desired, desiredMask, nybbleCheckPos, data);

    if (kthLargestIsPositive) {
      // Iterate in greatest -> least order (we want larger positive values)
      for (int i = kNybbles - 1; i >= 0; --i) {
        CHECK_NYBBLE();
      }
    } else {
      // Iterate in least -> greatest order (we want smaller negative
      // values)
      for (int i = 0; i < kNybbles; ++i) {
        CHECK_NYBBLE();
      }
    }

    if (nybbleCheckPos < 0) {
      // We have scanned all nybbles, and haven't found a unique
      // result. Therefore, there is a non-unique result that matches
      // the bit pattern 'desired' entirely; return it.
      return detail::findAnswer(data, desired, ~0U, kToFind);
    }
  }

#undef CHECK_NYBBLE
}

} // detail

/**
   Finds the Kth highest floating point value in a linear array [arr,
   end) without modifying the data and without temporary storage except
   for registers.

   - K starts at 1.
   - All threads in the warp will return the value.
   - Handles all floats except NaNs.
   - This function minimizes warp divergence.

   Returns the number of times the top-Kth element uniquely occurs
   along with its value.
*/
__device__ Pair<float, int>
warpFindTopKthElement(const DeviceTensor<float, 1>& data, int k) {
  if (data.getSize(0) <= WARP_SIZE) {
    // We can do this with a single warp coherent sort
    return detail::warpFindTopKthElementSmall32(data, k);
  } else {
    return detail::warpFindTopKthElementLarge(data, k);
  }
}

} } // namespace
