// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <cuda_runtime.h>

namespace facebook { namespace cuda {

/// Utility class for timing execution of a kernel
class KernelTimer {
 public:
  /// Constructor starts the timer and adds an event into the current
  /// device stream
  KernelTimer();

  /// Destructor releases event resources
  ~KernelTimer();

  /// Adds a stop event then synchronizes on the stop event to get the
  /// actual GPU-side kernel timings for any kernels launched in the
  /// current stream. Returns the number of milliseconds elapsed
  float stop();

 private:
  cudaEvent_t startEvent_;
  cudaEvent_t stopEvent_;
};

} } // namespace
