// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "cuda/DeviceTensor.cuh"

using namespace facebook::cuda;

namespace facebook { namespace cuda {

__device__ inline bool inBounds(
  int x, int padL, const DeviceTensor<float, 2>& t) {
  // Rely on unsigned integer arithmetic to test both 0 <= and < t.getSize()
  // in one shot.
  return ((unsigned)(x - padL) < (unsigned)(t.getSize(1)));
}


__device__ inline bool inBounds(
  int y, int x, int padU, int padL, const DeviceTensor<float, 3>& t) {
  // Rely on unsigned integer arithmetic to test both 0 <= and < t.getSize()
  // in one shot.
  return ((unsigned)(y - padU) < (unsigned)(t.getSize(1)) &&
          (unsigned)(x - padL) < (unsigned)(t.getSize(2)));
}

__device__ __forceinline__ bool inBounds(
  int y, int x, int padU, int padL, const DeviceTensor<float, 4>& t) {
  // Rely on unsigned integer arithmetic to test both 0 <= and < t.getSize()
  // in one shot.
  return ((unsigned)(y - padU) < (unsigned)(t.getSize(2)) &&
          (unsigned)(x - padL) < (unsigned)(t.getSize(3)));
}

__device__ inline bool inBounds(
  int x, const DeviceTensor<float, 2>& t) {
  // Rely on unsigned integer arithmetic to test both 0 <= and < t.getSize()
  // in one shot.
  return ((unsigned)(x) < (unsigned)(t.getSize(1)));
}

__device__ inline bool inBounds(int y, int x, const DeviceTensor<float, 3>& t) {
  // Rely on unsigned integer arithmetic to test both 0 <= and < t.getSize()
  // in one shot.
  return ((unsigned)(y) < (unsigned)(t.getSize(1)) &&
          (unsigned)(x) < (unsigned)(t.getSize(2)));
}

__device__ inline bool inBounds(int y, int x, const DeviceTensor<float, 4>& t) {
  // Rely on unsigned integer arithmetic to test both 0 <= and < t.getSize()
  // in one shot.
  return ((unsigned)(y) < (unsigned)(t.getSize(2)) &&
          (unsigned)(x) < (unsigned)(t.getSize(3)));
}

}} // ns
