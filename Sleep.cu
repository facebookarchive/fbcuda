#include <cstdio>
#include <cuda_runtime_api.h>
#include <mutex>
// FIXME: nvcc and gcc 4.9 don't like std::unordered_map
#include <tr1/unordered_map>

#include "util/CachedDeviceProperties.h"

namespace facebook { namespace cuda {

namespace {

constexpr int kDebug = false;

}

template <int type>
__global__ void sleepKernel(double* cycles, int64_t waitCycles) {
  extern __shared__ int s[];
  long long int start = clock64();
  for (;;) {
    auto total = clock64() - start;
    if (total >= waitCycles) { break; }
  }
  *cycles = (double(clock64() - start));
}

void cudaSleep(int64_t cycles, int type) {
  static std::mutex m;
  std::lock_guard<std::mutex> _(m);

  int device;
  cudaGetDevice(&device);

  static std::tr1::unordered_map<int, double*> clocks;
  auto e = clocks.find(device);
  if (e == clocks.end()) {
    double* c;
    cudaMalloc((void**)(&c), sizeof(double));
    clocks[device] = c;
    e = clocks.find(device);
  }
  auto t = e->second;

  auto p = getCurrentDeviceProperties();
  int minGridSize, blockSize;
  cudaOccupancyMaxPotentialBlockSize(
    &minGridSize, &blockSize, sleepKernel<1>, 0, 0);
  // Use all available SMs, smem to force kernel to eat up all resources
#define SLEEP(TYPE)                                                     \
  if (type == TYPE) {                                                   \
    sleepKernel<TYPE><<<minGridSize, blockSize>>> (t, cycles);          \
    double tt;                                                          \
    cudaMemcpy(&tt, t, sizeof(double), cudaMemcpyDeviceToHost);         \
    if (kDebug) {                                                       \
      unsigned long micros = (tt / p.clockRate) * 1000;                 \
      printf("cuda slept %ld us\n", micros);                            \
    }                                                                   \
    return;                                                             \
  }

  SLEEP(1);
  SLEEP(2);
  SLEEP(3);
  SLEEP(4);
  SLEEP(5);

  sleepKernel<99><<<minGridSize, blockSize>>>(t, cycles);
  double tt;
  cudaMemcpy(&tt, t, sizeof(double), cudaMemcpyDeviceToHost);
  if (kDebug) {
    unsigned long micros = (tt / p.clockRate) * 1000;
    printf("cuda slept %ld us\n", micros);
  }
}

} }

extern "C" void cudaSleepFFI(int64_t cycles, int type) {
  facebook::cuda::cudaSleep(cycles, type);
}
