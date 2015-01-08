// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/ComputeCapabilities.cuh"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "cuda/ShuffleTypes.cuh"

namespace facebook { namespace cuda {

/**
   `cuComplex` wrapper.
*/
struct Complex {
  __host__ __device__ __forceinline__
  Complex() {}

  __host__ __device__ __forceinline__
  Complex(float re) : cplx_(make_cuComplex(re, 0.0f)) {}

  __host__ __device__ __forceinline__
  Complex(float re, float im) : cplx_(make_cuComplex(re, im)) {}

  __host__ __device__ __forceinline__
  Complex(const Complex& c) : cplx_(c.cplx_) {}

  __host__ __device__ __forceinline__
  Complex(const cuComplex& c) : cplx_(c) {}

  __host__ __device__ __forceinline__
  Complex& operator=(const Complex& c) {
    // No need for swap
    cplx_ = c.cplx_;
    return *this;
  }

  __host__ __device__ __forceinline__
  bool operator==(const Complex& c) const {
    return cplx_.x == c.cplx_.x && cplx_.y == c.cplx_.y;
  }

  __host__ __device__ __forceinline__
  bool operator!=(const Complex& c) const {
    return !operator==(c);
  }

  __host__ __device__ __forceinline__
  Complex operator-() const {
    return Complex(make_cuComplex(-cplx_.x, -cplx_.y));
  }

  __host__ __device__ __forceinline__
  Complex operator-(const Complex& c) const {
    return Complex(cuCsubf(cplx_, c.cplx_));
  }

  __host__ __device__ __forceinline__
  Complex operator+(const Complex& c) const {
    return Complex(cuCaddf(cplx_, c.cplx_));
  }

  __host__ __device__ __forceinline__
  Complex operator*(const Complex& c) const {
    return Complex(cuCmulf(cplx_, c.cplx_));
  }

  __host__ __device__ __forceinline__
  Complex operator/(const Complex& c) const {
    return Complex(cuCdivf(cplx_, c.cplx_));
  }

  __host__ __device__ __forceinline__
  Complex& operator+=(const Complex& c) {
    cplx_ = cuCaddf(cplx_, c.cplx_);
    return *this;
  }

  __host__ __device__ __forceinline__
  Complex& operator-=(const Complex& c) {
    cplx_ = cuCsubf(cplx_, c.cplx_);
    return *this;
  }

  __host__ __device__ __forceinline__
  Complex& operator*=(const Complex& c) {
    cplx_ = cuCmulf(cplx_, c.cplx_);
    return *this;
  }

  __host__ __device__ __forceinline__
  Complex& operator/=(const Complex& c) {
    cplx_ = cuCdivf(cplx_, c.cplx_);
    return *this;
  }

  __host__ __device__ __forceinline__
  Complex transpose() const {
    return Complex(make_cuComplex(cplx_.y, cplx_.x));
  }

  __host__ __device__ __forceinline__
  Complex conjugate() const {
    return Complex(make_cuComplex(cplx_.x, -cplx_.y));
  }

  __host__ __device__ __forceinline__
  void cexp(float angle) {
    sincosf(angle, &cplx_.y, &cplx_.x);
  }

  __host__ __device__ __forceinline__
  float& re() {
    return cplx_.x;
  }

  __host__ __device__ __forceinline__
  float& im() {
    return cplx_.y;
  }

  __host__ __device__ __forceinline__
  const float& re() const {
    return cplx_.x;
  }

  __host__ __device__ __forceinline__
  const float& im() const {
    return cplx_.y;
  }

  __host__ __device__ __forceinline__
  operator float2() const {
    return static_cast<float2>(cplx_);
  }

private:
  cuComplex cplx_;
};

#ifdef __CUDA_ARCH__

__device__ __forceinline__ Complex
shfl(const Complex& c, int lane, int bound = WARP_SIZE) {
  return Complex(shfl(c.re(), lane, bound),
                 shfl(c.im(), lane, bound));
}

__device__ __forceinline__ Complex
shfl_up(const Complex& c, int lane, int bound = WARP_SIZE) {
  return Complex(shfl_up(c.re(), lane, bound),
                 shfl_up(c.im(), lane, bound));
}

__device__ __forceinline__ Complex
shfl_down(const Complex& c, int lane, int bound = WARP_SIZE) {
  return Complex(shfl_down(c.re(), lane, bound),
                 shfl_down(c.im(), lane, bound));
}

__device__ __forceinline__ Complex
shfl_xor(const Complex& c, int lane, int bound = WARP_SIZE) {
  return Complex(shfl_xor(c.re(), lane, bound),
                 shfl_xor(c.im(), lane, bound));
}

#endif

}}
