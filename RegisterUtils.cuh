// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/ComputeCapabilities.cuh"
#include "cuda/CudaStaticAssert.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/Pair.cuh"
#include "cuda/ShuffleTypes.cuh"

#include <boost/preprocessor/repetition/repeat.hpp>

namespace facebook { namespace cuda {

/// Various utilities for dealing with arrays of values which are
/// maintained in thread-local registers. All accesses are done in such
/// a way such that the index is statically known, which preserves the
/// compiler's ability to allocate the values to registers, as opposed
/// to local memory.
template <typename T, int N>
struct RegisterUtils {
  /// Register shifting: move elements towards the beginning of the
  /// array (towards 0) by `Shift` places:
  /// arr[i] = arr[i + Shift]
  /// The `Shift` elements at the end are left unchanged.
  template <int Shift>
  __device__ __forceinline__ static void shiftLeft(T arr[N]) {
    // e.g., N = 5, Shift = 2:
    // 0 1 2 3 4 becomes =>
    // 2 3 4 3 4 (last are unchanged)
#pragma unroll
    for (int i = 0; i < N - Shift; ++i) {
      arr[i] = arr[i + Shift];
    }
  }

  /// Register shifting: move elements towards the end of the
  /// array (towards N - 1) by `Shift` places:
  /// arr[i] = arr[i - Shift]
  /// The `Shift` elements at the beginning are left unchanged.
  template <int Shift>
  __device__ __forceinline__ static void shiftRight(T arr[N]) {
    // e.g., N = 5, Shift = 2:
    // 0 1 2 3 4 becomes =>
    // 0 1 0 1 2 (first are unchanged)
#pragma unroll
    for (int i = N - 1; i >= Shift; --i) {
      arr[i] = arr[i - Shift];
    }
  }

  /// Register rotation: move elements cyclically towards the beginning
  /// of the array with wrap around (towards 0).
  template <int Rotate>
  __device__ __forceinline__ static void rotateLeft(T arr[N]) {
    T tmp[Rotate];
    // e.g., N = 5, Rotate = 2:
    // 0 1 2 3 4 becomes =>
    // 2 3 4 0 1

    // copy 0 1
#pragma unroll
    for (int i = 0; i < Rotate; ++i) {
      tmp[i] = arr[i];
    }

    // 0 1 2 x x =>
    // 2 3 4 x x
#pragma unroll
    for (int i = 0; i < N - Rotate; ++i) {
      arr[i] = arr[i + Rotate];
    }

    // x x x 3 4 =>
    // x x x 0 1
#pragma unroll
    for (int i = 0; i < Rotate; ++i) {
      arr[N - Rotate + i] = tmp[i];
    }
  }

  /// Register rotation: move elements cyclically towards the end
  /// of the array with wrap around (towards N - 1).
  template <int Rotate>
  __device__ __forceinline__ static void rotateRight(T arr[N]) {
    T tmp[Rotate];
    // e.g., N = 5, Rotate = 2:
    // 0 1 2 3 4 becomes =>
    // 3 4 0 1 2

    // copy 3 4
#pragma unroll
    for (int i = 0; i < Rotate; ++i) {
      tmp[i] = arr[N - Rotate + i];
    }

    // x x 2 3 4 =>
    // x x 0 1 2
#pragma unroll
    for (int i = N - 1; i >= Rotate; --i) {
      arr[i] = arr[i - Rotate];
    }

    // 0 1 x x x =>
    // 3 4 x x x
#pragma unroll
    for (int i = 0; i < Rotate; ++i) {
      arr[i] = tmp[i];
    }
  }
};

/**
   
   Utilities for addressing values held in register arrays, but with a
   dynamic index. For instance, if you had:

   ~~~{.cpp}
   float arr[6];
   int index = calculation();
   arr[index + 1] = doStuffWith(arr[index]);
   ~~~
   
   the dynamic indexing of `arr` with `index` requires that the
   compiler address `arr` in local memory, not registers, removing any
   performance benefit.
   Usually one should use static indexing for register arrays, for
   example:

   ~~~{.cpp}
   #pragma unroll
   for (int i = 0; i < 6; ++i) { arr[i] = foo; }
   ~~~
   
   or

   ~~~{.cpp}
   arr[3] = foo;
   ~~~
   
   in order to allow the compiler to assign registers to `arr`, but
   there are occasions when one needs to dynamically index the array.
   The arrays in question should often be very small (e.g., N = 2-3)
   to avoid any lookup penalty.

   These utilities translate the dynamic request to a static request,
   for array sizes N = 1 to 32.

   So, to take our original case, you'd use it like:

   ~~~{.cpp}
   float arr[6];
   int index = calculation();
   float val = doStuffWith(RegisterUtils<float, 6>::get(arr, index));
   RegisterUtils<float, 6>::set(arr, index + 1, val);
   ~~~
   
   which will preserve the compiler's ability to assign `arr` to
   registers.
*/
template <typename T, int N>
struct RegisterIndexUtils {
  /// Retrieve a single value from our thread-local register array
  __device__ __forceinline__ static T get(const T arr[N], int index);

  /// Set a single value in our thread-local register array
  __device__ __forceinline__ static void set(T arr[N], int index, T val);
};

/// Utilities for warp-wide held register arrays
template <typename T, int N>
struct WarpRegisterUtils {
  /// Broadcast a single value from the warp-wide array `arr`,
  /// considering `index` as an index across the warp threads.
  /// In other words, returns arr[index / warpSize] from lane (index %
  /// warpSize) to all threads in the warp.
  __device__ static T broadcast(const T arr[N], int index) {
    // Figure out which lane
    const int lane = index & (WARP_SIZE - 1);
    const int bucket = index / WARP_SIZE;

    return shfl(RegisterIndexUtils<T, N>::get(arr, bucket), lane);
  }
};

/// Tensor <-> register load/save utils, for managing a set of
/// registers distributed across the warp
template <typename T, int N>
struct WarpRegisterLoaderUtils {
  /// Convenience utility to load values from a 1-d array into
  /// registers using within-warp striding.
  /// Registers for which there is no entry in the array get `fillVal`
  /// as a value
  __device__ static void load(T arr[N],
                              const DeviceTensor<T, 1>& in,
                              const T fill) {
    const int lane = getLaneId();

    for (int i = 0; i < N; ++i) {
      const int offset = lane + i * WARP_SIZE;
      arr[i] = (offset < in.getSize(0)) ? in[offset] : fill;
    }
  }

  /// Convenience utility to save values into a 1-d array from
  /// registers using within-warp striding.
  /// Saves up to `num` values from the registers.
  __device__ static void save(DeviceTensor<T, 1>& out,
                              const T arr[N],
                              const int num) {
    const int lane = getLaneId();

    for (int i = 0; i < N; ++i) {
      const int offset = lane + i * WARP_SIZE;
      if (offset < num) {
        out[offset] = arr[i];
      }
    }
  }
};

/// Tensor <-> register load/save utils for Pair<>, for managing a set
/// of registers distributed across the warp
template <typename K, typename V, int N>
struct WarpRegisterPairLoaderUtils {
  /// Like WarpRegisterUtils<T>::load, but for key/value pair
  /// types. Initializes the value with the source index.
  __device__ static void load(Pair<K, V> arr[N],
                              const DeviceTensor<K, 1>& in,
                              const K keyFill,
                              const V valueFill) {
    const int lane = getLaneId();

    for (int i = 0; i < N; ++i) {
      const int offset = lane + i * WARP_SIZE;
      arr[i] = (offset < in.getSize(0)) ?
        Pair<K, V>(in[offset], offset) : Pair<K, V>(keyFill, valueFill);
    }
  }

  /// Like WarpRegisterUtils<T>::load, but for key/value pair
  /// types. The value for each key is at the corresponding index in
  /// the value array. The arrays are presumed to be the same size.
  __device__ static void load(Pair<K, V> arr[N],
                              const DeviceTensor<K, 1>& key,
                              const DeviceTensor<V, 1>& value,
                              const K keyFill,
                              const V valueFill) {
    const int lane = getLaneId();

    for (int i = 0; i < N; ++i) {
      const int offset = lane + i * WARP_SIZE;
      arr[i] = (offset < key.getSize(0)) ?
        Pair<K, V>(key[offset], value[offset]) : Pair<K, V>(keyFill, valueFill);
    }
  }

  /// Like WarpRegisterUtils<T>::save, but for key/value pair types.
  __device__ static void save(DeviceTensor<K, 1>& key,
                              DeviceTensor<V, 1>& value,
                              const Pair<K, V> arr[N],
                              const int num) {
    const int lane = getLaneId();

    for (int i = 0; i < N; ++i) {
      const int offset = lane + i * WARP_SIZE;

      if (offset < num) {
        key[offset] = arr[i].k;
        value[offset] = arr[i].v;
      }
    }
  }
};

#define GET_CASE(UNUSED1, I, UNUSED2)           \
  case I:                                       \
  return arr[I];

#define SET_CASE(UNUSED1, I, UNUSED2)           \
  case I:                                       \
  arr[I] = val;                                 \
  break;

#define IMPL_REGISTER_ARRAY(N)                                          \
template <typename T>                                                   \
  struct RegisterIndexUtils<T, N> {                                     \
  __device__ __forceinline__ static T get(const T arr[N], int index) {  \
    switch (index) {                                                    \
      BOOST_PP_REPEAT(N, GET_CASE, 0);                                  \
      default:                                                          \
        return T();                                                     \
    };                                                                  \
  }                                                                     \
                                                                        \
  __device__ __forceinline__ static void set(T arr[N], int index, T val) { \
    switch (index) {                                                    \
      BOOST_PP_REPEAT(N, SET_CASE, 0);                                  \
    }                                                                   \
  }                                                                     \
};

#define IMPL_REGISTER_ARRAY_CASE(UNUSED1, I, UNUSED2) IMPL_REGISTER_ARRAY(I);

BOOST_PP_REPEAT(32, IMPL_REGISTER_ARRAY_CASE, 0);

} } // namespace
