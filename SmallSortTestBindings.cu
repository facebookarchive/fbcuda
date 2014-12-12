// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/SmallSortTestBindings.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/SmallSort.cuh"

using namespace std;

namespace facebook { namespace cuda {

__global__ void
sortDevice(DeviceTensor<float, 1> data, DeviceTensor<float, 1> out) {
  warpSort<float, GreaterThan<float> >(data, out);
}

__global__ void
sortDevice(DeviceTensor<float, 1> data,
           DeviceTensor<float, 1> out,
           DeviceTensor<int, 1> indices) {
  warpSort<float, int, GreaterThan<Pair<float, int> > >(data, out, indices);
}

vector<float>
sort(const vector<float>& data) {
  const size_t sizeBytes = data.size() * sizeof(float);

  float* devFloat = NULL;
  cudaMalloc(&devFloat, sizeBytes);
  cudaMemcpy(devFloat, data.data(), sizeBytes, cudaMemcpyHostToDevice);

  float* devResult = NULL;
  cudaMalloc(&devResult, sizeBytes);
  cudaMemset(devResult, 0, sizeBytes);

  dim3 grid(1);
  dim3 block(32);

  int dataSizes[] = { (int) data.size() };
  int outSizes[] = { (int) data.size() };

  sortDevice<<<grid, block>>>(
    DeviceTensor<float, 1>(devFloat, dataSizes),
    DeviceTensor<float, 1>(devResult, outSizes));

  vector<float> vals(data.size());
  cudaMemcpy(vals.data(), devResult, sizeBytes, cudaMemcpyDeviceToHost);

  cudaFree(devFloat);
  cudaFree(devResult);

  return vals;
}

vector<pair<float, int> >
sortWithIndices(const std::vector<float>& data) {
  const size_t sizeBytes = data.size() * sizeof(float);
  const size_t sizeIndicesBytes = data.size() * sizeof(int);

  float* devFloat = NULL;
  cudaMalloc(&devFloat, sizeBytes);
  cudaMemcpy(devFloat, data.data(), sizeBytes, cudaMemcpyHostToDevice);

  float* devResult = NULL;
  cudaMalloc(&devResult, sizeBytes);
  cudaMemset(devResult, 0, sizeBytes);
  int* devIndices = NULL;
  cudaMalloc(&devIndices, sizeIndicesBytes);
  cudaMemset(devIndices, 0, sizeIndicesBytes);

  dim3 grid(1);
  dim3 block(32);

  int dataSizes[] = { (int) data.size() };
  int outSizes[] = { (int) data.size() };

  sortDevice<<<grid, block>>>(
    DeviceTensor<float, 1>(devFloat, dataSizes),
    DeviceTensor<float, 1>(devResult, outSizes),
    DeviceTensor<int, 1>(devIndices, outSizes));

  vector<float> vals(data.size());
  cudaMemcpy(vals.data(),
             devResult, sizeBytes, cudaMemcpyDeviceToHost);

  vector<int> indices(data.size());
  cudaMemcpy(indices.data(),
             devIndices, sizeIndicesBytes, cudaMemcpyDeviceToHost);

  cudaFree(devFloat);
  cudaFree(devResult);
  cudaFree(devIndices);

  vector<pair<float, int> > result;
  for (int i = 0; i < data.size(); ++i) {
    result.push_back(make_pair(vals[i], indices[i]));
  }

  return result;
}

} } // namespace
