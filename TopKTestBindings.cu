// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/TopKTestBindings.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/TopK.cuh"
#include "cuda/TopKElements.cuh"

using namespace std;

namespace facebook { namespace cuda {

__device__ float topKthElementAnswer;

__global__ void
findTopKthElementDevice(DeviceTensor<float, 1> data, int k) {
  float topK = warpFindTopKthElement(data, k).k;

  if (threadIdx.x == 0) {
    topKthElementAnswer = topK;
  }
}

__global__ void
findTopKElementsValueOrderDevice(DeviceTensor<float, 1> data,
                                 DeviceTensor<float, 1> out,
                                 int k) {
  warpFindTopKElementsValueOrder(data, out, k);
}

__global__ void
findTopKElementsIndexOrderDevice(DeviceTensor<float, 1> data,
                                 DeviceTensor<float, 1> out,
                                 DeviceTensor<int, 1> indices,
                                 int k) {
  warpFindTopKElementsIndexOrder(data, out, indices, k);
}

__global__ void
findTopKElementsValueOrderDevice(DeviceTensor<float, 1> data,
                                 DeviceTensor<float, 1> out,
                                 DeviceTensor<int, 1> indices,
                                 int k) {
  warpFindTopKElementsValueOrder(data, out, indices, k);
}

float
findTopKthElement(const vector<float>& data, int k) {
  const size_t sizeBytes = data.size() * sizeof(float);
  float* devFloat = NULL;
  cudaMalloc(&devFloat, sizeBytes);
  cudaMemcpy(devFloat, data.data(), sizeBytes, cudaMemcpyHostToDevice);

  dim3 grid(1);
  dim3 block(32);

  int dataSizes[] = { (int) data.size() };

  findTopKthElementDevice<<<grid, block>>>(
    DeviceTensor<float, 1>(devFloat, dataSizes), k);

  float val = 0.0f;
  cudaMemcpyFromSymbol(
    &val, topKthElementAnswer, sizeof(float), 0, cudaMemcpyDeviceToHost);

  cudaFree(devFloat);

  return val;
}

vector<float>
findTopKElements(const vector<float>& data, int k) {
  const size_t sizeBytes = data.size() * sizeof(float);
  const size_t sizeResultBytes = k * sizeof(float);

  float* devFloat = NULL;
  cudaMalloc(&devFloat, sizeBytes);
  cudaMemcpy(devFloat, data.data(), sizeBytes, cudaMemcpyHostToDevice);

  float* devResult = NULL;
  cudaMalloc(&devResult, sizeResultBytes);
  cudaMemset(devResult, 0, sizeResultBytes);

  dim3 grid(1);
  dim3 block(32);

  int dataSizes[] = { (int) data.size() };
  int outSizes[] = { k };

  findTopKElementsValueOrderDevice<<<grid, block>>>(
    DeviceTensor<float, 1>(devFloat, dataSizes),
    DeviceTensor<float, 1>(devResult, outSizes),
    k);

  vector<float> vals(k);
  cudaMemcpy(vals.data(), devResult, sizeResultBytes, cudaMemcpyDeviceToHost);

  cudaFree(devFloat);
  cudaFree(devResult);

  return vals;
}

vector<pair<float, int> >
findTopKElementsAndIndicesIndexOrder(const std::vector<float>& data, int k) {
  const size_t sizeBytes = data.size() * sizeof(float);
  const size_t sizeResultBytes = k * sizeof(float);
  const size_t sizeIndicesBytes = k * sizeof(int);

  float* devFloat = NULL;
  cudaMalloc(&devFloat, sizeBytes);
  cudaMemcpy(devFloat, data.data(), sizeBytes, cudaMemcpyHostToDevice);

  float* devResult = NULL;
  cudaMalloc(&devResult, sizeResultBytes);
  cudaMemset(devResult, 0, sizeResultBytes);
  int* devIndices = NULL;
  cudaMalloc(&devIndices, sizeIndicesBytes);
  cudaMemset(devIndices, 0, sizeIndicesBytes);

  dim3 grid(1);
  dim3 block(32);

  int dataSizes[] = { (int) data.size() };
  int outSizes[] = { k };

  findTopKElementsIndexOrderDevice<<<grid, block>>>(
    DeviceTensor<float, 1>(devFloat, dataSizes),
    DeviceTensor<float, 1>(devResult, outSizes),
    DeviceTensor<int, 1>(devIndices, outSizes),
    k);

  vector<float> vals(k);
  cudaMemcpy(vals.data(),
             devResult, sizeResultBytes, cudaMemcpyDeviceToHost);

  vector<int> indices(k);
  cudaMemcpy(indices.data(),
             devIndices, sizeIndicesBytes, cudaMemcpyDeviceToHost);

  cudaFree(devFloat);
  cudaFree(devResult);
  cudaFree(devIndices);

  vector<pair<float, int> > result;
  for (int i = 0; i < k; ++i) {
    result.push_back(make_pair(vals[i], indices[i]));
  }

  return result;
}


vector<pair<float, int> >
findTopKElementsAndIndicesValueOrder(const std::vector<float>& data, int k) {
  const size_t sizeBytes = data.size() * sizeof(float);
  const size_t sizeResultBytes = k * sizeof(float);
  const size_t sizeIndicesBytes = k * sizeof(int);

  float* devFloat = NULL;
  cudaMalloc(&devFloat, sizeBytes);
  cudaMemcpy(devFloat, data.data(), sizeBytes, cudaMemcpyHostToDevice);

  float* devResult = NULL;
  cudaMalloc(&devResult, sizeResultBytes);
  cudaMemset(devResult, 0, sizeResultBytes);
  int* devIndices = NULL;
  cudaMalloc(&devIndices, sizeIndicesBytes);
  cudaMemset(devIndices, 0, sizeIndicesBytes);

  dim3 grid(1);
  dim3 block(32);

  int dataSizes[] = { (int) data.size() };
  int outSizes[] = { k };

  findTopKElementsValueOrderDevice<<<grid, block>>>(
    DeviceTensor<float, 1>(devFloat, dataSizes),
    DeviceTensor<float, 1>(devResult, outSizes),
    DeviceTensor<int, 1>(devIndices, outSizes),
    k);

  vector<float> vals(k);
  cudaMemcpy(vals.data(),
             devResult, sizeResultBytes, cudaMemcpyDeviceToHost);

  vector<int> indices(k);
  cudaMemcpy(indices.data(),
             devIndices, sizeIndicesBytes, cudaMemcpyDeviceToHost);

  cudaFree(devFloat);
  cudaFree(devResult);
  cudaFree(devIndices);

  vector<pair<float, int> > result;
  for (int i = 0; i < k; ++i) {
    result.push_back(make_pair(vals[i], indices[i]));
  }

  return result;
}

} } // namespace
