// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

/**
   A nvcc-compilable version of static_assert. Remove once their
   compiler achieves C++11.
*/
template <bool>
struct CudaStaticAssert;

template <>
struct CudaStaticAssert<true> {
};

#define cuda_static_assert(expr) \
  (CudaStaticAssert<(expr) != 0>())
