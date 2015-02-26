// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <float.h>

/** @file
    Contains utilities useful in creating deterministic parallel
    floating-point reductions.
 */

namespace facebook { namespace cuda {

/// Constructs a rounding factor used to truncate elements in a sum
/// such that the sum of the truncated elements is the same no matter
/// what the order of the sum is.
///
/// Floating point summation is not associative; using this factor
/// makes it associative, so a parallel sum can be performed in any
/// order (presumably using atomics).
///
/// Follows Algorithm 5: Reproducible Sequential Sum in
/// 'Fast Reproducible Floating-Point Summation' by Demmel and Nguyen
/// http://www.eecs.berkeley.edu/~hdnguyen/public/papers/ARITH21_Fast_Sum.pdf
///
/// For summing x_i, i = 1 to n:
/// @param max The maximum seen floating point value abs(x_i)
/// @param n The number of elements for the sum, or an upper bound estimate
__host__ __device__ inline float
createRoundingFactor(float max, int n) {
  float delta = (max * n) / (1.0f - 2 * n * FLT_EPSILON);

  // Calculate ceil(log_2(delta)).
  // frexpf() calculates exp and returns `x` such that
  // delta = x * 2^exp, where `x` in (-1.0, -0.5] U [0.5, 1).
  // Because |x| < 1, exp is exactly ceil(log_2(delta)).
  int exp;
  frexpf(delta, &exp);

  // return M = 2 ^ ceil(log_2(delta))
  return ldexpf(1.0f, exp);
}

/// Given the rounding factor in `createRoundingFactor` calculated
/// using max(|x_i|), truncate `x` to a value that can be used for a
/// deterministic, reproducible parallel sum of all x_i.
__host__ __device__ inline float
truncateWithRoundingFactor(float roundingFactor, float x) {
  return (roundingFactor + x) - roundingFactor;
}

} } // namespace
