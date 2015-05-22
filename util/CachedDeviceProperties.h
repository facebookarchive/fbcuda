// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#pragma once

#include <cuda_runtime.h>

namespace facebook { namespace cuda {

int getDevice();

const cudaDeviceProp& getCurrentDeviceProperties();
const cudaDeviceProp& getDeviceProperties(int device);

} }
