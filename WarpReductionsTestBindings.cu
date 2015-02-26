// Copyright 2004-present Facebook. All Rights Reserved.
#include "cuda/WarpReductionsTestBindings.cuh"
#include "cuda/WarpReductions.cuh"

#include <cuda.h>
#include <stdio.h>

using namespace std;

namespace facebook { namespace cuda {

__device__ int hasDuplicate[32];

__global__ void checkDuplicates(int num, int* v) {
  hasDuplicate[threadIdx.x] = (int) warpHasCollision(v[threadIdx.x]);
}

__device__ unsigned int duplicateMask;

__global__ void checkDuplicateMask(int num, int* v) {
  unsigned int mask = warpCollisionMask(v[threadIdx.x]);
  if (threadIdx.x == 0) {
    duplicateMask = mask;
  }
}

vector<int> hostCheckDuplicates(const vector<int>& v) {
  int* devSet = NULL;
  cudaMalloc(&devSet, v.size() * sizeof(int));
  cudaMemcpy(devSet, v.data(), v.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  checkDuplicates<<<1, 32>>>(v.size(), devSet);

  vector<int> hasDuplicates(32, false);
  cudaMemcpyFromSymbol(hasDuplicates.data(),
                       hasDuplicate, sizeof(int) * 32, 0,
                       cudaMemcpyDeviceToHost);
  cudaFree(devSet);

  return hasDuplicates;
}

unsigned int hostCheckDuplicateMask(const vector<int>& v) {
  int* devSet = NULL;
  cudaMalloc(&devSet, v.size() * sizeof(int));
  cudaMemcpy(devSet, v.data(), v.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  checkDuplicateMask<<<1, 32>>>(v.size(), devSet);

  unsigned int mask = 0;
  cudaMemcpyFromSymbol(&mask,
                       duplicateMask, sizeof(unsigned int), 0,
                       cudaMemcpyDeviceToHost);
  cudaFree(devSet);

  return mask;
}

} }
