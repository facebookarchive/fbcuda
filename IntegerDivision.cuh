// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/CudaUtils.cuh"
#include <math.h>

namespace facebook { namespace cuda {

namespace detail {

/**
   Host and device implementation for 32-bit a * b into 64 bit, return
   high 32 bits
*/
__host__ __device__ __forceinline__ unsigned int mulHi(unsigned int x,
                                                       unsigned int y) {
#ifdef __CUDA_ARCH__
  return __umulhi(x, y);
#else
  unsigned long v = (unsigned long) x * (unsigned long) y;
  return (unsigned) (v >> 32);
#endif
}

} // detail

/**
   Prototype for integer division by a fixed constant via strength
   reduction to mul/shift.
*/
template <typename T>
class FixedDivisor {
};

/**
   Specialization for calculating quotients by a fixed signed `d`
   using integer multiplication and shifts.
*/
template <>
class FixedDivisor<int> {
 public:
  typedef int Type;

  FixedDivisor(int d)
      : d_(d) {
    calcSignedMagic();
  }

  /// Calculates `q = n / d`.
  __host__ __device__ __forceinline__ int div(int n) {
    return (detail::mulHi(magic_, n) >> shift_);
  }

  /// Calculates `r = n % d`.
  __host__ __device__ __forceinline__ int mod(int n) {
    return n - d_ * div(n);
  }

  /// Calculates `q = n / d` and `r = n % d` together.
  __host__ __device__ __forceinline__
  void divMod(int n, int* q, int* r) {
    const int quotient = div(n);
    *q = quotient;
    *r = n - d_ * quotient;
  }

 private:
  /**
     Calculates magic multiplicative value and shift amount for
     calculating `q = n / d` for signed 32-bit integers.
     Implementation taken from Hacker's Delight section 10.
     `d` cannot be in [-1, 1].
  */
  void calcSignedMagic() {
    const unsigned int two31 = 0x80000000;

    unsigned int ad = abs(d_);
    unsigned int t = two31 + ((unsigned int) d_ >> 31);
    unsigned int anc = t - 1 - t % ad;   // Absolute value of nc.
    unsigned int p = 31;                 // Init. p.
    unsigned int q1 = two31 / anc;       // Init. q1 = 2**p/|nc|.
    unsigned int r1 = two31 - q1 * anc;  // Init. r1 = rem(2**p, |nc|).
    unsigned int q2 = two31 / ad;        // Init. q2 = 2**p/|d|.
    unsigned int r2 = two31 - q2 * ad;   // Init. r2 = rem(2**p, |d|).
    unsigned int delta = 0;

    do {
      p = p + 1;
      q1 = 2 * q1;         // Update q1 = 2**p/|nc|.
      r1 = 2 * r1;         // Update r1 = rem(2**p, |nc|).

      if (r1 >= anc) {     // (Must be an unsigned
        q1 = q1 + 1;       // comparison here).
        r1 = r1 - anc;
      }

      q2 = 2 * q2;         // Update q2 = 2**p/|d|.
      r2 = 2 * r2;         // Update r2 = rem(2**p, |d|).

      if (r2 >= ad) {      // (Must be an unsigned
        q2 = q2 + 1;       // comparison here).
        r2 = r2 - ad;
      }

      delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));

    magic_ = q2 + 1;
    if (d_ < 0) {
      magic_ = -magic_;
    }
    shift_ = p - 32;
  }

  int d_;
  int magic_;
  int shift_;
};

/**
   Class for calculating quotients by a fixed unsigned `d` using integer
   multiplication, addition and shifts.
*/
template <>
class FixedDivisor<unsigned int> {
 public:
  typedef unsigned int Type;

  FixedDivisor(unsigned int d)
      : d_(d) {
    calcUnsignedMagic();
  }

  /// Calculates `q = n / d`.
  __host__ __device__ __forceinline__ unsigned int div(unsigned int n) {
    if (add_) {
      // Calculates (t + q) / 2, avoiding overflow
      unsigned int q = detail::mulHi(magic_, n);
      unsigned int t = n - q;
      t >>= 1;
      t += q;

      // Shift (t + q) / 2
      return (t >> (shift_));

    } else {
      return (detail::mulHi(magic_, n) >> shift_);
    }
  }

  /// Calculates `r = n % d`.
  __host__ __device__ __forceinline__ unsigned int mod(unsigned int n) {
    return n - d_ * div(n);
  }

  /// Calculates `q = n / d` and `r = n % d` together.
  __host__ __device__ __forceinline__
  void divMod(unsigned int n, unsigned int* q, unsigned int* r) {
    const unsigned int quotient = div(n);
    *q = quotient;
    *r = n - d_ * quotient;
  }

 private:
  /**
     Calculates magic multiplicative value, addition indicator and shift
     amount for calculating `q = n / d` for unsigned 32-bit
     integers. Implementation taken from Hacker's Delight section 10.
     `d` must be > 1.
  */
  void calcUnsignedMagic() {
    int gt = 0;
    unsigned int nc = (unsigned int) -1 - (unsigned int) (-(int) d_) % d_;
    int p = 31;                 // Init. p.
    unsigned int q1 = 0x80000000 / nc;       // Init. q1 = 2**p/nc.
    unsigned int r1 = 0x80000000 - q1 * nc;  // Init. r1 = rem(2**p, nc).
    unsigned int q2 = 0x7FFFFFFF / d_;       // Init. q2 = (2**p - 1)/d.
    unsigned int r2 = 0x7FFFFFFF - q2 * d_;  // Init. r2 = rem(2**p - 1, d).
    unsigned int delta = 0;
    add_ = false;

    do {
      p = p + 1;

      if (q1 >= 0x80000000) {
        gt = 1;  // Means q1 > delta.
      }

      if (r1 >= nc - r1) {
        q1 = 2 * q1 + 1;  // Update q1.
        r1 = 2 * r1 - nc; // Update r1.
      } else {
        q1 = 2 * q1;
        r1 = 2 * r1;
      }

      if (r2 + 1 >= d_ - r2) {
        if (q2 >= 0x7FFFFFFF) {
          add_ = true;
        }
        q2 = 2 * q2 + 1;     // Update q2.
        r2 = 2 * r2 + 1 - d_; // Update r2.
      } else {
        if (q2 >= 0x80000000) {
          add_ = true;
        }
        q2 = 2 * q2;
        r2 = 2 * r2 + 1;
      }

      delta = d_ - 1 - r2;
    } while (gt == 0 &&
             (q1 < delta || (q1 == delta && r1 == 0)));

    magic_ = q2 + 1;
    shift_ = p - 32;

    // We combine the division together for the output
    if (add_) {
      --shift_;
    }
  }

  unsigned int d_;
  unsigned int magic_;
  bool add_;
  int shift_;
};

} }
