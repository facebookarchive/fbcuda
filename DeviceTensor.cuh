// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/CudaStaticAssert.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

namespace facebook { namespace cuda {

/// Our tensor type
template <typename T, int Dim>
class DeviceTensor;

/// Type of a subspace of a tensor
namespace detail {
template <typename TensorType, int SubDim>
class DeviceSubTensor;
}

/**
   Templated multi-dimensional array that supports strided access of
   elements. Main access is through `operator[]`; e.g.,
   `tensor[x][y][z]`.

- `T` is the contained type (e.g., `float`)
- `Dim` is the tensor rank
*/
template <typename T, int Dim>
class DeviceTensor {
 public:
  enum { NumDim = Dim };
  typedef T DataType;
  typedef DeviceTensor<T, Dim> TensorType;

  /// Default constructor
  __host__ __device__ DeviceTensor();

  /// Constructor that calculates strides with no padding
  __host__ __device__ DeviceTensor(T* data, const int sizes[Dim]);

  /// Constructor that takes arbitrary size/stride arrays
  __host__ __device__ DeviceTensor(T* data,
                                   const int sizes[Dim],
                                   const int strides[Dim]);

  /// Returns true if the two tensors are of the same dimensionality,
  /// size and stride.
  template <int OtherDim>
  __host__ __device__ bool
  isSameSizeAndStride(const DeviceTensor<T, OtherDim>& rhs) const;

  /// Produces a string containing our size and stride array contents;
  /// for debugging purposes
  std::string toString() const;

  /// Cast to a tensor of a different type of the same size and stride
  template <typename U>
  __host__ __device__ DeviceTensor<U, Dim> cast();

  template <typename U>
  __host__ __device__ const DeviceTensor<U, Dim> cast() const;

  /// Returns a raw pointer to the start of our data.
  __host__ __device__ __forceinline__ typename TensorType::DataType* data() {
    return data_;
  }

  /// Returns a raw pointer to the start of our data (const).
  __host__ __device__ __forceinline__
  const typename TensorType::DataType* data() const {
    return data_;
  }

  /// Cast to a different datatype
  template <typename U>
  __host__ __device__ __forceinline__ U* dataAs() {
    return reinterpret_cast<U*>(data_);
  }

  /// Cast to a different datatype
  template <typename U>
  __host__ __device__ __forceinline__ const U* dataAs() const {
    return reinterpret_cast<U*>(data_);
  }

  /// Returns a read/write view of a portion of our tensor.
  __host__ __device__ __forceinline__
  detail::DeviceSubTensor<TensorType, Dim - 1> operator[](int);

  /// Returns a read/write view of a portion of our tensor (const).
  __host__ __device__ __forceinline__
  const detail::DeviceSubTensor<TensorType, Dim - 1> operator[](int) const;

  /// Returns the size of a given dimension, `[0, Dim - 1]`. No bounds checking.
  __host__ __device__ __forceinline__ int getSize(int i) const {
    return size_[i];
  }

  /// Returns the size of a given dimension, `[0, Dim - 1]`. No bounds checking.
  __host__ __device__ __forceinline__ int getStride(int i) const {
    return stride_[i];
  }

  /// Returns the total number of elements contained within our data
  /// (product of `getSize(i)`)
  __host__ __device__ long numElements() const;

  /// Returns the size array.
  __host__ __device__ __forceinline__ const int* sizes() const {
    return size_;
  }

  /// Returns the stride array.
  __host__ __device__ __forceinline__ const int* strides() const {
    return stride_;
  }

  /// Limited form of resize by permutation, make sure your permutation array
  /// is legit. Only works for contiguous tensors.
  void permuteDims(const std::vector<int>& perm);

  /// Returns true if there is no padding within the tensor and no
  /// re-ordering of the dimensions.
  /// ~~~
  /// (stride(i) == size(i + 1) * stride(i + 1))
  /// ~~~
  __host__ __device__ bool isContiguous() const;

  /// Returns whether a given dimension has only increasing stride
  /// from the previous dimension. A tensor that was permuted by
  /// exchanging size and stride only will fail this check.
  /// If `i == 0` just check `size > 0`. Returns `false` if `stride` is `<= 0`.
  __host__ __device__ bool isConsistentlySized(int i) const;

  // Returns whether at each dimension `stride <= size`.
  // If this is not the case then iterating once over the size space will
  // touch the same memory locations multiple times.
  __host__ __device__ bool isConsistentlySized() const;

  /// Returns true if the given dimension index has no padding
  __host__ __device__ bool isContiguousDim(int i) const;

  /// Upcast a tensor of dimension `D` to some tensor of dimension
  /// D' > D by padding the leading dimensions by 1
  /// e.g., upcasting a 2-d tensor `[2][3]` to a 4-d tensor `[1][1][2][3]`
  template <int NewDim>
  __host__ __device__ DeviceTensor<T, NewDim> upcastOuter();

  /// Upcast a tensor of dimension `D` to some tensor of dimension
  /// D' > D by padding the lowest/most varying dimensions by 1
  /// e.g., upcasting a 2-d tensor `[2][3]` to a 4-d tensor `[2][3][1][1]`
  template <int NewDim>
  __host__ __device__ DeviceTensor<T, NewDim> upcastInner();

  /// Downcast a tensor of dimension `D` to some tensor of dimension
  /// D' < D by collapsing the leading dimensions. asserts if there is
  /// padding on the leading dimensions.
  template <int NewDim>
  __host__ __device__ DeviceTensor<T, NewDim> downcastOuter();

  /// Downcast a tensor of dimension `D` to some tensor of dimension
  /// D' < D by collapsing the leading dimensions. asserts if there is
  /// padding on the leading dimensions.
  template <int NewDim>
  __host__ __device__ DeviceTensor<T, NewDim> downcastInner();

  /// Returns a tensor that is a view of the `SubDim`-dimensional slice
  /// of this tensor, starting at `at`.
  template <int SubDim>
  __host__ __device__ DeviceTensor<T, SubDim> view(T* at);

  /// Returns a tensor that is a view of the `SubDim`-dimensional slice
  /// of this tensor, starting where our data begins
  template <int SubDim>
  __host__ __device__ DeviceTensor<T, SubDim> view();

  /// Fill a value into our data asynchronously.
  void fillAsync(T val, cudaStream_t stream = 0);

 private:
  /// Raw pointer to where the tensor data begins
  T* data_;

  /// Array of strides (in sizeof(T) terms) per each dimension
  int stride_[Dim];

  /// Size per each dimension
  int size_[Dim];
};

namespace detail {

/// Specialization for a view of a single value (0-dimensional)
template <typename TensorType>
class DeviceSubTensor<TensorType, 0> {
 public:
  __host__ __device__ DeviceSubTensor<TensorType, 0>
  operator=(typename TensorType::DataType val) {
    *data_ = val;
    return *this;
  }

  // operator T&
  __host__ __device__ operator typename TensorType::DataType&() {
    return *data_;
  }

  // const operator T& returning const T&
  __host__ __device__ operator const typename TensorType::DataType&() const {
    return *data_;
  }

  // operator& returning T*
  __host__ __device__ typename TensorType::DataType* operator&() {
    return data_;
  }

  // const operator& returning const T*
  __host__ __device__ const typename TensorType::DataType* operator&() const {
    return data_;
  }

  /// Returns a raw accessor to our slice.
  __host__ __device__ __forceinline__ typename TensorType::DataType* data() {
    return data_;
  }

  /// Returns a raw accessor to our slice (const).
  __host__ __device__ __forceinline__
  const typename TensorType::DataType* data() const {
    return data_;
  }

  /// Cast to a different datatype.
  template <typename T>
  __host__ __device__ T& as() {
    return *dataAs<T>();
  }

  /// Cast to a different datatype (const).
  template <typename T>
  __host__ __device__ const T& as() const {
    return *dataAs<T>();
  }

  /// Cast to a different datatype
  template <typename T>
  __host__ __device__ __forceinline__ T* dataAs() {
    return reinterpret_cast<T*>(data_);
  }

  /// Cast to a different datatype (const)
  template <typename T>
  __host__ __device__ __forceinline__ const T* dataAs() const {
    return reinterpret_cast<T*>(data_);
  }

  /// Use the texture cache for reads
  __device__ __forceinline__ typename TensorType::DataType ldg() const {
    return __ldg(const_cast<const typename TensorType::DataType*>(data_));
  }

  /// Use the texture cache for reads; cast as a particular type
  template <typename T>
  __device__ __forceinline__ T ldgAs() const {
    return __ldg(dataAs<T>());
  }

  private:
  /// One dimension greater can create us
  friend class DeviceSubTensor<TensorType, 1>;

  /// Our parent tensor can create us
  friend class DeviceTensor<typename TensorType::DataType, 1>;

  __host__ __device__ __forceinline__ DeviceSubTensor(
    TensorType& t,
    typename TensorType::DataType* data)
      : tensor_(t),
        data_(data) {
  }

  /// The tensor we're referencing
  TensorType& tensor_;

  /// Where our value is located
  typename TensorType::DataType* const data_;
};

/// A `SubDim`-rank slice of a parent DeviceTensor
template <typename TensorType, int SubDim>
class DeviceSubTensor {
 public:
  /// Returns a view of the data located at our offset (the dimension `SubDim` - 1
  /// tensor).
  __host__ __device__ __forceinline__
  DeviceSubTensor<TensorType, SubDim - 1> operator[](int index) {
    return DeviceSubTensor<TensorType, SubDim - 1>(
      tensor_,
      data_ + index * tensor_.getStride(TensorType::NumDim - SubDim));
  }

  /// Returns a view of the data located at our offset (the dimension `SubDim` - 1
  /// tensor) (const).
  __host__ __device__ __forceinline__
  const DeviceSubTensor<TensorType, SubDim - 1> operator[](int index) const {
    return DeviceSubTensor<TensorType, SubDim - 1>(
      tensor_,
      data_ + index * tensor_.getStride(TensorType::NumDim - SubDim));
  }

  // operator& returning T*
  __host__ __device__ typename TensorType::DataType* operator&() {
    return data_;
  }

  // const operator& returning const T*
  __host__ __device__ const typename TensorType::DataType* operator&() const {
    return data_;
  }

  /// Returns a raw accessor to our slice.
  __host__ __device__ __forceinline__ typename TensorType::DataType* data() {
    return data_;
  }

  /// Returns a raw accessor to our slice (const).
  __host__ __device__ __forceinline__
  const typename TensorType::DataType* data() const {
    return data_;
  }

  /// Cast to a different datatype
  template <typename T>
  __host__ __device__ __forceinline__ T* dataAs() {
    return reinterpret_cast<T*>(data_);
  }

  /// Cast to a different datatype (const)
  template <typename T>
  __host__ __device__ __forceinline__ const T* dataAs() const {
    return reinterpret_cast<T*>(data_);
  }

  /// Use the texture cache for reads
  __device__ __forceinline__ typename TensorType::DataType ldg() const {
    return __ldg(const_cast<const typename TensorType::DataType*>(data_));
  }

  /// Use the texture cache for reads; cast as a particular type
  template <typename T>
  __device__ __forceinline__ T ldgAs() const {
    return __ldg(dataAs<T>());
  }

  /// Returns a tensor that is a view of the SubDim-dimensional slice
  /// of this tensor, starting where our data begins
  DeviceTensor<typename TensorType::DataType, SubDim> view() {
    return tensor_.template view<SubDim>(data_);
  }

 private:
  /// One dimension greater can create us
  friend class DeviceSubTensor<TensorType, SubDim + 1>;

  /// Our parent tensor can create us
  friend class
  DeviceTensor<typename TensorType::DataType, TensorType::NumDim>;

  __host__ __device__ __forceinline__ DeviceSubTensor(
    TensorType& t,
    typename TensorType::DataType* data)
      : tensor_(t),
        data_(data) {
  }

  /// The tensor we're referencing
  TensorType& tensor_;

  /// The start of our sub-region
  typename TensorType::DataType* const data_;
};

} // namespace detail

template <typename T, int Dim>
__host__ __device__ __forceinline__
detail::DeviceSubTensor<DeviceTensor<T, Dim>, Dim - 1>
DeviceTensor<T, Dim>::operator[](int index) {
  return detail::DeviceSubTensor<TensorType, Dim - 1>(
    detail::DeviceSubTensor<TensorType, Dim>(*this, data_)[index]);
}

template <typename T, int Dim>
__host__ __device__ __forceinline__
const detail::DeviceSubTensor<DeviceTensor<T, Dim>, Dim - 1>
DeviceTensor<T, Dim>::operator[](int index) const {
  return detail::DeviceSubTensor<TensorType, Dim - 1>(
    detail::DeviceSubTensor<TensorType, Dim>(
      const_cast<TensorType&>(*this), data_)[index]);
}

/// Streaming operator for logging
template <typename T, int Dim>
std::ostream& operator<<(std::ostream& os, const DeviceTensor<T, Dim>& t) {
  os << t.toString();
  return os;
}

} } // namespace

#include "cuda/DeviceTensor-inl.cuh"
