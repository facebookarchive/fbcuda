#include "cuda/KernelTimer.h"

namespace facebook { namespace cuda {

KernelTimer::KernelTimer()
    : startEvent_(0),
      stopEvent_(0) {
  cudaEventCreate(&startEvent_);
  cudaEventCreate(&stopEvent_);

  cudaEventRecord(startEvent_, 0);
}

KernelTimer::~KernelTimer() {
  cudaEventDestroy(startEvent_);
  cudaEventDestroy(stopEvent_);
}

float
KernelTimer::stop() {
  cudaEventRecord(stopEvent_, 0);
  cudaEventSynchronize(stopEvent_);

  auto time = 0.0f;
  cudaEventElapsedTime(&time, startEvent_, stopEvent_);
  return time;
}

} }
