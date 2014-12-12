// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <assert.h>
#include <sstream>

#ifndef __CUDA_ARCH__
// host code
#include <stdexcept>
#endif

namespace facebook { namespace cuda {

namespace detail {

template <int N>
__host__ __device__ void copy(int to[N], int from[N]) {
  for (int i = 0; i < N; ++i) {
    to[i] = from[i];
  }
}

} // namespace detail

template <typename T, int Dim>
__host__ __device__
DeviceTensor<T, Dim>::DeviceTensor()
    : data_(NULL) {
  cuda_static_assert(Dim > 0);

  for (int i = 0; i < Dim; ++i) {
    size_[i] = 0;
    stride_[i] = 1;
  }
}

template <typename T, int Dim>
__host__ __device__
DeviceTensor<T, Dim>::DeviceTensor(T* data, const int sizes[Dim])
    : data_(data) {
  cuda_static_assert(Dim > 0);

  for (int i = 0; i < Dim; ++i) {
    size_[i] = sizes[i];
  }

  stride_[Dim - 1] = 1;
  for (int i = Dim - 2; i >= 0; --i) {
    stride_[i] = stride_[i + 1] * sizes[i + 1];
  }
}

template <typename T, int Dim>
__host__ __device__
DeviceTensor<T, Dim>::DeviceTensor(T* data,
                                   const int sizes[Dim],
                                   const int strides[Dim])
    : data_(data) {
  cuda_static_assert(Dim > 0);

  for (int i = 0; i < Dim; ++i) {
    size_[i] = sizes[i];
    stride_[i] = strides[i];
  }
}

template <typename T, int Dim>
template <int OtherDim>
__host__ __device__ bool
DeviceTensor<T, Dim>::isSameSizeAndStride(
  const DeviceTensor<T, OtherDim>& rhs) const {
  if (Dim != OtherDim) {
    return false;
  }

  for (int i = 0; i < Dim; ++i) {
    if (size_[i] != rhs.size_[i]) {
      return false;
    }

    if (stride_[i] != rhs.stride_[i]) {
      return false;
    }
  }

  return true;
}

template <typename T, int Dim>
std::string
DeviceTensor<T, Dim>::toString() const {
  // FIXME: convert to folly::format once CUDA is C++11-capable
  std::stringstream ss;

  ss << "sizes: [";
  for (int i = 0; i < Dim; ++i) {
    ss << getSize(i);

    if (i < Dim - 1) {
      ss << ", ";
    }
  }

  ss << "]  strides: [";

  for (int i = 0; i < Dim; ++i) {
    ss << getStride(i);

    if (i < Dim - 1) {
      ss << ", ";
    }
  }

  ss << "]";
  return ss.str();
}

template <typename T, int Dim>
template <typename U>
__host__ __device__ DeviceTensor<U, Dim>
DeviceTensor<T, Dim>::cast() {
  cuda_static_assert(sizeof(U) == sizeof(T));

  return DeviceTensor<U, Dim>(reinterpret_cast<U*>(data_),
                              size_,
                              stride_);
}

template <typename T, int Dim>
template <typename U>
__host__ __device__ const DeviceTensor<U, Dim>
DeviceTensor<T, Dim>::cast() const {
  cuda_static_assert(sizeof(U) == sizeof(T));

  return DeviceTensor<U, Dim>(reinterpret_cast<U*>(data_),
                              size_,
                              stride_);
}

template <typename T, int Dim>
long
DeviceTensor<T, Dim>::numElements() const {
  long size = getSize(0);

  for (int i = 1; i < Dim; ++i) {
    size *= getSize(i);
  }

  return size;
}

template <typename T, int Dim>
void
DeviceTensor<T, Dim>::permuteDims(const std::vector<int>& perm) {
  // This only works for contiguous tensors since strides are
  // recomputed
#ifndef __CUDA_ARCH__
  if (perm.size() != Dim) {
    throw std::invalid_argument("Permutation list must be of the same size "
                                "as our dimension");
  }

  if (getStride(Dim - 1) != 1) {
    throw std::invalid_argument("Innermost dimension must have stride 1");
  }

  for (int i = 0; i < Dim; ++i) {
    if (!isContiguousDim(i)) {
      throw std::invalid_argument("All dimensions must be contiguous");
    }
  }
#endif

  // Permute
  int newSizes[Dim];
  int newStrides[Dim];
  for (int i = 0; i < Dim; ++i) {
    newSizes[i] = getSize(perm[i]);
  }

  newStrides[Dim - 1] = 1;
  for (int i = Dim - 2; i >= 0; --i) {
    newStrides[i] = newStrides[i + 1] * newSizes[i + 1];
  }

  detail::copy<Dim>(size_, newSizes);
  detail::copy<Dim>(stride_, newStrides);

  // Output sanity check
  for (int i = 0; i < Dim; ++i) {
    assert(isContiguousDim(i));
  }
}

template <typename T, int Dim>
__host__ __device__ bool
DeviceTensor<T, Dim>::isContiguous() const {
  long prevSize = 1;

  for (int i = Dim - 1; i >= 0; --i) {
    if (getSize(i) != 1) {
      if (getStride(i) == prevSize) {
        prevSize *= getSize(i);
      } else {
        return false;
      }
    }
  }

  return true;
}

template <typename T, int Dim>
__host__ __device__ bool
DeviceTensor<T, Dim>::isConsistentlySized(int i) const {
  if (i == 0 && getStride(i) > 0 && getSize(i) > 0) {
    return true;
  } else if ((i > 0) && (i < Dim) && (getStride(i) > 0) &&
             ((getStride(i - 1) / getStride(i)) >= getSize(i))) {
    return true;
  }

  return false;
}

template <typename T, int Dim>
__host__ __device__ bool
DeviceTensor<T, Dim>::isConsistentlySized() const {
  for (int i = 0; i < Dim; ++i) {
    if (!isConsistentlySized(i)) {
      return false;
    }
  }

  return true;
}

template <typename T, int Dim>
__host__ __device__ bool
DeviceTensor<T, Dim>::isContiguousDim(int i) const {
  return (i == Dim - 1) || // just in case
    ((i < Dim - 1) &&
     ((getStride(i) / getStride(i + 1)) == getSize(i + 1)));
}

template <typename T, int Dim>
template <int NewDim>
__host__ __device__ DeviceTensor<T, NewDim>
DeviceTensor<T, Dim>::upcastOuter() {
  // Can only create tensors of greater dimension
  cuda_static_assert(NewDim > Dim);

  int newSize[NewDim];
  int newStride[NewDim];

  int shift = NewDim - Dim;

  for (int i = 0; i < NewDim; ++i) {
    if (i < shift) {
      // These are the extended dimensions
      newSize[i] = 1;
      newStride[i] = size_[0] * stride_[0];
    } else {
      // Shift the remaining dimensions
      newSize[i] = size_[i - shift];
      newStride[i] = stride_[i - shift];
    }
  }

  return DeviceTensor<T, NewDim>(data_, newSize, newStride);
}

template <typename T, int Dim>
template <int NewDim>
__host__ __device__ DeviceTensor<T, NewDim>
DeviceTensor<T, Dim>::upcastInner() {
  // Can only create tensors of greater dimension
  cuda_static_assert(NewDim > Dim);

  int newSize[NewDim];
  int newStride[NewDim];

  for (int i = 0; i < NewDim; ++i) {
    if (i < Dim) {
      // Existing dimensions get copied over
      newSize[i] = size_[i];
      newStride[i] = stride_[i];
    } else {
      // Extended dimensions
      newSize[i] = 1;
      newStride[i] = 1;
    }
  }

  return DeviceTensor<T, NewDim>(data_, newSize, newStride);
}

template <typename T, int Dim>
template <int NewDim>
__host__ __device__ DeviceTensor<T, NewDim>
DeviceTensor<T, Dim>::downcast() {
  // Can only create tensors of lesser dimension
  cuda_static_assert(NewDim < Dim);

  // We can't downcast non-contiguous tensors, since it leaves
  // garbage data in the tensor. The tensor needs to be contiguous
  // in all of the dimensions we are collapsing (no padding in
  // them).
  for (int i = 0; i < Dim - NewDim; ++i) {
    bool cont = isContiguousDim(i);
#ifdef __CUDA_ARCH__
    // Device code
    assert(isContiguousDim(i));
#else
    // Host code
    if (!cont) {
      throw std::invalid_argument("Can only downcast contiguous tensors");
    }
#endif
  }

  int newSize[NewDim];
  int newStride[NewDim];

  int ignoredDims = Dim - NewDim;
  int collapsedSize = 1;

  for (int i = 0; i < Dim; ++i) {
    if (i < ignoredDims) {
      // Collapse these dimensions
      collapsedSize *= getSize(i);
    } else {
      // Non-collapsed dimensions
      if (i == ignoredDims) {
        // This is the first non-collapsed dimension
        newSize[i - ignoredDims] = collapsedSize * getSize(i);
      } else {
        // Subsequent non-collapsed dimensions
        newSize[i - ignoredDims] = getSize(i);
      }

      newStride[i - ignoredDims] = getStride(i);
    }
  }

  return DeviceTensor<T, NewDim>(data_, newSize, newStride);
}

template <typename T, int Dim>
template <int SubDim>
__host__ __device__ DeviceTensor<T, SubDim>
DeviceTensor<T, Dim>::view(T* at) {
  cuda_static_assert(SubDim >= 1 && SubDim < Dim);

  int viewSizes[SubDim];
  int viewStrides[SubDim];

  for (int i = 0; i < SubDim; ++i) {
    viewSizes[i] = size_[Dim - SubDim + i];
    viewStrides[i] = stride_[Dim - SubDim + i];
  }

  return DeviceTensor<T, SubDim>(at, viewSizes, viewStrides);
}

template <typename T, int Dim>
template <int SubDim>
__host__ __device__ DeviceTensor<T, SubDim>
DeviceTensor<T, Dim>::view() {
  return view<SubDim>(data_);
}

template <typename T, int Dim>
void
DeviceTensor<T, Dim>::fillAsync(T val, cudaStream_t stream) {
  // Can only use cudaMemsetAsync with `int` sized types
  cuda_static_assert(sizeof(T) == sizeof(int));

#ifndef __CUDA_ARCH__
  if (!isContiguous()) {
    throw std::invalid_argument("fillAsync only works on contiguous data");
  }
#endif

  union {
    T vT;
    int vInt;
  } u;

  u.vT = val;

  cudaMemsetAsync(data(), u.vInt, numElements() * sizeof(T), stream);
}

} } // namespace
