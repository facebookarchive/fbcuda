// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include "cuda/util/CachedDeviceProperties.h"

#include <cassert>
#include <memory>

using namespace std;

namespace facebook { namespace cuda {

namespace {

void checkCuda(cudaError_t error, string&& s) {
  if (error) {
    std::string err(cudaGetErrorString(error));
    throw std::runtime_error(s + err);
  }
}

struct DeviceProperties {
  DeviceProperties();
  int deviceCount = 0;
  std::unique_ptr<cudaDeviceProp[]> deviceProperties;
};

DeviceProperties::DeviceProperties() {
  auto err = cudaGetDeviceCount(&deviceCount);
  if (err == cudaErrorNoDevice) {
    deviceCount = 0;
  } else {
    checkCuda(err, std::string("CUDA ERROR: cudaGetDeviceCount "));
  }

  deviceProperties = std::unique_ptr<cudaDeviceProp[]>(
    new cudaDeviceProp[deviceCount]);
  for (int i = 0; i < deviceCount; ++i) {
    auto err = cudaGetDeviceProperties(&deviceProperties[i], i);
    checkCuda(err, std::string("CUDA ERROR: cudaGetDeviceCount "));
  }
}

}  // namespace

int getDevice() {
  int dev;
  checkCuda(cudaGetDevice(&dev), std::string("CUDA ERROR: cudaGetDevice "));
  return dev;
}

const cudaDeviceProp& getCurrentDeviceProperties() {
  int device = 0;
  auto err = cudaGetDevice(&device);
  checkCuda(err, std::string("CUDA ERROR: cudaGetDeviceCount "));

  return getDeviceProperties(device);
}

const cudaDeviceProp& getDeviceProperties(int device) {
  // Thread-safe initialization guaranteed by C++11 memory model
  static DeviceProperties dprop;
  assert(device >= 0 && device < dprop.deviceCount);
  return dprop.deviceProperties[device];
}

} }
