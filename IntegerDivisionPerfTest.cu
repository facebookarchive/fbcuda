// Copyright 2004-present Facebook. All Rights Reserved.
#include "cuda/IntegerDivision.cuh"

#include <stdio.h>

using namespace std;
using namespace facebook::cuda;

#define kNumBlocks 1200
#define kNumThreadsPerBlock 32 * 4
#define kNumElemPerThread 64

__device__ int getStartOffset() {
  return blockIdx.x * kNumThreadsPerBlock * kNumElemPerThread +
    threadIdx.x;
}

template <typename T>
__global__ void divideFixed(int num, T* data, T div) {
  int sum = 0;

  for (int i = 0; i < kNumElemPerThread; ++i) {
    T v = data[i * kNumThreadsPerBlock + getStartOffset()];
    T d = v / div;
    T r = v % div;

    sum += d + r;
  }

  data[getStartOffset()] = sum;
}

template <typename T>
__global__ void divideMagicFixed(int num, T* data, FixedDivisor<T> div) {
  T sum = 0;

  for (int i = 0; i < kNumElemPerThread; ++i) {
    T v = data[i * kNumThreadsPerBlock + getStartOffset()];
    T d, r;
    div.divMod(v, &d, &r);

    sum += d + r;
  }

  data[getStartOffset()] = sum;
}

int main(int argc, char** argv) {
  int d = 55;
  int num = kNumBlocks * kNumThreadsPerBlock * kNumElemPerThread;

  unsigned int* dev = NULL;
  cudaMalloc(&dev, num * sizeof(unsigned int));
  cudaMemset(dev, 5, num * sizeof(unsigned int));

  divideMagicFixed<int><<<kNumBlocks, kNumThreadsPerBlock>>>(
    num, (int*) dev, FixedDivisor<int>(d));
  cudaMemset(dev, 5, num * sizeof(unsigned int));

  divideMagicFixed<unsigned int><<<kNumBlocks, kNumThreadsPerBlock>>>(
    num, dev, FixedDivisor<unsigned int>(d));
  cudaMemset(dev, 5, num * sizeof(unsigned int));

  divideFixed<int><<<kNumBlocks, kNumThreadsPerBlock>>>(num, (int*) dev, d);
  cudaMemset(dev, 5, num * sizeof(unsigned int));

  divideFixed<unsigned int><<<kNumBlocks, kNumThreadsPerBlock>>>(num, dev, d);

  cudaDeviceSynchronize();
  cudaFree(dev);
}
