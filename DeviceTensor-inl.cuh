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

template <typename T, int N>
__host__ __device__ void copy(T to[N], T from[N]) {
  for (int i = 0; i < N; ++i) {
    to[i] = from[i];
  }
}

} // namespace detail

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
DeviceTensor<T, Dim, IndexT, PtrTraits>::DeviceTensor()
    : data_(NULL) {
  cuda_static_assert(Dim > 0);

  for (int i = 0; i < Dim; ++i) {
    size_[i] = 0;
    stride_[i] = (IndexT) 1;
  }
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
DeviceTensor<T, Dim, IndexT, PtrTraits>::
DeviceTensor(DataPtrType data, const IndexT sizes[Dim])
    : data_(data) {
  cuda_static_assert(Dim > 0);

  for (int i = 0; i < Dim; ++i) {
    size_[i] = sizes[i];
  }

  stride_[Dim - 1] = (IndexT) 1;
  for (int i = Dim - 2; i >= 0; --i) {
    stride_[i] = stride_[i + 1] * sizes[i + 1];
  }
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
DeviceTensor<T, Dim, IndexT, PtrTraits>::DeviceTensor(DataPtrType data,
                                                      const IndexT sizes[Dim],
                                                      const IndexT strides[Dim])
    : data_(data) {
  cuda_static_assert(Dim > 0);

  for (int i = 0; i < Dim; ++i) {
    size_[i] = sizes[i];
    stride_[i] = strides[i];
  }
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int OtherDim>
__host__ __device__ bool
DeviceTensor<T, Dim, IndexT, PtrTraits>::isSameSizeAndStride(
  const DeviceTensor<T, OtherDim, IndexT, PtrTraits>& rhs) const {
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

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
std::string
DeviceTensor<T, Dim, IndexT, PtrTraits>::toString() const {
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

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <typename U>
__host__ __device__ DeviceTensor<U, Dim, IndexT, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::cast() {
  cuda_static_assert(sizeof(U) == sizeof(T));

  return DeviceTensor<U, Dim, IndexT, PtrTraits>(
    reinterpret_cast<U*>(data_), size_, stride_);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <typename U>
__host__ __device__ const DeviceTensor<U, Dim, IndexT, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::cast() const {
  cuda_static_assert(sizeof(U) == sizeof(T));

  return DeviceTensor<U, Dim, IndexT, PtrTraits>(reinterpret_cast<U*>(data_),
                                                 size_,
                                                 stride_);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ long
DeviceTensor<T, Dim, IndexT, PtrTraits>::numElements() const {
  long size = getSize(0);

  for (int i = 1; i < Dim; ++i) {
    size *= getSize(i);
  }

  return size;
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
void
DeviceTensor<T, Dim, IndexT, PtrTraits>::
permuteDims(const std::vector<int>& perm) {
  // This only works for contiguous tensors since strides are
  // recomputed
#ifndef __CUDA_ARCH__
  if (perm.size() != Dim) {
    throw std::invalid_argument("Permutation list must be of the same size "
                                "as our dimension");
  }

  if (getStride(Dim - 1) != (IndexT) 1) {
    throw std::invalid_argument("Innermost dimension must have stride 1");
  }

  for (int i = 0; i < Dim; ++i) {
    if (!isContiguousDim(i)) {
      throw std::invalid_argument("All dimensions must be contiguous");
    }
  }
#endif

  // Permute
  IndexT newSizes[Dim];
  IndexT newStrides[Dim];
  for (int i = 0; i < Dim; ++i) {
    newSizes[i] = getSize(perm[i]);
  }

  newStrides[Dim - 1] = 1;
  for (int i = Dim - 2; i >= 0; --i) {
    newStrides[i] = newStrides[i + 1] * newSizes[i + 1];
  }

  detail::copy<IndexT, Dim>(size_, newSizes);
  detail::copy<IndexT, Dim>(stride_, newStrides);

  // Output sanity check
  for (int i = 0; i < Dim; ++i) {
    assert(isContiguousDim(i));
  }
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ bool
DeviceTensor<T, Dim, IndexT, PtrTraits>::isContiguous() const {
  long prevSize = 1;

  for (int i = Dim - 1; i >= 0; --i) {
    if (getSize(i) != (IndexT) 1) {
      if (getStride(i) == prevSize) {
        prevSize *= getSize(i);
      } else {
        return false;
      }
    }
  }

  return true;
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ bool
DeviceTensor<T, Dim, IndexT, PtrTraits>::isConsistentlySized(int i) const {
  if (i == 0 && getStride(i) > 0 && getSize(i) > 0) {
    return true;
  } else if ((i > 0) && (i < Dim) && (getStride(i) > 0) &&
             ((getStride(i - 1) / getStride(i)) >= getSize(i))) {
    return true;
  }

  return false;
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ bool
DeviceTensor<T, Dim, IndexT, PtrTraits>::isConsistentlySized() const {
  for (int i = 0; i < Dim; ++i) {
    if (!isConsistentlySized(i)) {
      return false;
    }
  }

  return true;
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ bool
DeviceTensor<T, Dim, IndexT, PtrTraits>::isContiguousDim(int i) const {
  return (i == Dim - 1) || // just in case
    ((i < Dim - 1) &&
     ((getStride(i) / getStride(i + 1)) == getSize(i + 1)));
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ DeviceTensor<T, Dim, IndexT, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::transpose(int dim1, int dim2) const {
#ifdef __CUDA_ARCH__
  // Device code
  assert(dim1 >= 0 && dim1 < Dim);
  assert(dim1 >= 0 && dim2 < Dim);
#else
  // Host code
  if (dim1 < 0 || dim1 >= Dim) {
    throw std::invalid_argument("dim1 out of bounds");
  }

  if (dim2 < 0 || dim2 >= Dim) {
    throw std::invalid_argument("dim2 out of bounds");
  }
#endif

  IndexT newSize[Dim];
  IndexT newStride[Dim];

  for (int i = 0; i < Dim; ++i) {
    newSize[i] = size_[i];
    newStride[i] = stride_[i];
  }

  IndexT tmp = newSize[dim1];
  newSize[dim1] = newSize[dim2];
  newSize[dim2] = tmp;

  tmp = newStride[dim1];
  newStride[dim1] = newStride[dim2];
  newStride[dim2] = tmp;

  return DeviceTensor<T, Dim, IndexT, PtrTraits>(data_, newSize, newStride);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int NewDim>
__host__ __device__ DeviceTensor<T, NewDim, IndexT, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::upcastOuter() {
  // Can only create tensors of greater dimension
  cuda_static_assert(NewDim > Dim);

  IndexT newSize[NewDim];
  IndexT newStride[NewDim];

  int shift = NewDim - Dim;

  for (int i = 0; i < NewDim; ++i) {
    if (i < shift) {
      // These are the extended dimensions
      newSize[i] = (IndexT) 1;
      newStride[i] = size_[0] * stride_[0];
    } else {
      // Shift the remaining dimensions
      newSize[i] = size_[i - shift];
      newStride[i] = stride_[i - shift];
    }
  }

  return DeviceTensor<T, NewDim, IndexT, PtrTraits>(data_, newSize, newStride);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int NewDim>
__host__ __device__ DeviceTensor<T, NewDim, IndexT, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::upcastInner() {
  // Can only create tensors of greater dimension
  cuda_static_assert(NewDim > Dim);

  IndexT newSize[NewDim];
  IndexT newStride[NewDim];

  for (int i = 0; i < NewDim; ++i) {
    if (i < Dim) {
      // Existing dimensions get copied over
      newSize[i] = size_[i];
      newStride[i] = stride_[i];
    } else {
      // Extended dimensions
      newSize[i] = (IndexT) 1;
      newStride[i] = (IndexT) 1;
    }
  }

  return DeviceTensor<T, NewDim, IndexT, PtrTraits>(data_, newSize, newStride);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int NewDim>
__host__ __device__ DeviceTensor<T, NewDim, IndexT, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::downcastOuter() {
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
    assert(cont);
#else
    // Host code
    if (!cont) {
      throw std::invalid_argument("Can only downcast contiguous tensors");
    }
#endif
  }

  IndexT newSize[NewDim];
  IndexT newStride[NewDim];

  int ignoredDims = Dim - NewDim;
  IndexT collapsedSize = 1;

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

  return DeviceTensor<T, NewDim, IndexT, PtrTraits>(
    data_, newSize, newStride);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int NewDim>
__host__ __device__ DeviceTensor<T, NewDim, IndexT, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::downcastInner() {
  // Can only create tensors of lesser dimension
  cuda_static_assert(NewDim < Dim);

  // We can't downcast non-contiguous tensors, since it leaves
  // garbage data in the tensor. The tensor needs to be contiguous
  // in all of the dimensions we are collapsing (no padding in
  // them).
  for (int i = NewDim; i < Dim; ++i) {
    bool cont = isContiguousDim(i);
#ifdef __CUDA_ARCH__
    // Device code
    assert(cont);
#else
    // Host code
    if (!cont) {
      throw std::invalid_argument("Can only downcast contiguous tensors");
    }
#endif
  }

  IndexT newSize[NewDim];
  IndexT newStride[NewDim];

  IndexT collapsedSize = 1;

  for (int i = Dim - 1; i >= 0; --i) {
    if (i >= NewDim) {
      // Collapse these dimensions
      collapsedSize *= getSize(i);
    } else {
      // Non-collapsed dimensions
      if (i == NewDim - 1) {
        // This is the first non-collapsed dimension
        newSize[i] = collapsedSize * getSize(i);
        newStride[i] = getStride(Dim - 1);
      } else {
        // Subsequent non-collapsed dimensions
        newSize[i] = getSize(i);
        newStride[i] = getStride(i);
      }
    }
  }

  return DeviceTensor<T, NewDim, IndexT, PtrTraits>(data_, newSize, newStride);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int SubDim>
__host__ __device__ DeviceTensor<T, SubDim, IndexT, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::view(DataPtrType at) {
  cuda_static_assert(SubDim >= 1 && SubDim < Dim);

  IndexT viewSizes[SubDim];
  IndexT viewStrides[SubDim];

  for (int i = 0; i < SubDim; ++i) {
    viewSizes[i] = size_[Dim - SubDim + i];
    viewStrides[i] = stride_[Dim - SubDim + i];
  }

  return DeviceTensor<T, SubDim, IndexT, PtrTraits>(at, viewSizes, viewStrides);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int SubDim>
__host__ __device__ DeviceTensor<T, SubDim, IndexT, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::view() {
  return view<SubDim>(data_);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
void
DeviceTensor<T, Dim, IndexT, PtrTraits>::fillAsync(T val, cudaStream_t stream) {
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
