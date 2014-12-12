// Copyright 2004-present Facebook. All Rights Reserved.

#include <stdio.h> // for printf

namespace facebook { namespace cuda {

// Collection of CUDA debugging utilities

//
// PRINT_DEBUG
// These are macros since we need to concatenate fmt strings and only
// each printf() is atomic, and doing work in device land to
// concatenate strings is too much
//

#ifdef ENABLE_CUDA_DEBUG

#define PRINT_DEBUG(FMT, ...) {                                         \
  if (gridDim.y == 1 && gridDim.z == 1) {                               \
    if (blockDim.y == 1 && blockDim.z == 1) {                           \
      printf("(b%d t%d) " FMT "\n",                                     \
             blockIdx.x,                                                \
             threadIdx.x, __VA_ARGS__);                                 \
    } else if (blockDim.z == 1) {                                       \
      printf("(b%d t%d/%d) " FMT "\n",                                  \
             blockIdx.x,                                                \
             threadIdx.x, threadIdx.y, __VA_ARGS__);                    \
    } else {                                                            \
      printf("(b%d t%d/%d/%d) " FMT "\n",                               \
             blockIdx.x,                                                \
             threadIdx.x, threadIdx.y, threadIdx.z, __VA_ARGS__);       \
    }                                                                   \
  } else if (gridDim.z == 1) {                                          \
    if (blockDim.y == 1 && blockDim.z == 1) {                           \
      printf("(b%d/%d t%d) " FMT "\n",                                  \
             blockIdx.x, blockIdx.y,                                    \
             threadIdx.x, __VA_ARGS__);                                 \
    } else if (blockDim.z == 1) {                                       \
      printf("(b%d/%d t%d/%d) " FMT "\n",                               \
             blockIdx.x, blockIdx.y,                                    \
             threadIdx.x, threadIdx.y, __VA_ARGS__);                    \
    } else {                                                            \
      printf("(b%d/%d t%d/%d/%d) " FMT "\n",                            \
             blockIdx.x, blockIdx.y,                                    \
             threadIdx.x, threadIdx.y, threadIdx.z, __VA_ARGS__);       \
    }                                                                   \
  } else {                                                              \
    if (blockDim.y == 1 && blockDim.z == 1) {                           \
      printf("(b%d/%d/%d t%d) " FMT "\n",                               \
             blockIdx.x, blockIdx.y, blockIdx.z,                        \
             threadIdx.x, __VA_ARGS__);                                 \
    } else if (blockDim.z == 1) {                                       \
      printf("(b%d/%d/%d t%d/%d) " FMT "\n",                            \
             blockIdx.x, blockIdx.y, blockIdx.z,                        \
             threadIdx.x, threadIdx.y, __VA_ARGS__);                    \
    } else {                                                            \
      printf("(b%d/%d/%d t%d/%d/%d) " FMT "\n",                         \
             blockIdx.x, blockIdx.y, blockIdx.z,                        \
             threadIdx.x, threadIdx.y, threadIdx.z, __VA_ARGS__);       \
    }                                                                   \
  }                                                                     \
}

#define PRINT_DEBUG0(FMT) {                             \
  if (gridDim.y == 1 && gridDim.z == 1) {               \
    if (blockDim.y == 1 && blockDim.z == 1) {           \
      printf("(b%d t%d) " FMT "\n",                     \
             blockIdx.x,                                \
             threadIdx.x);                              \
    } else if (blockDim.z == 1) {                       \
      printf("(b%d t%d/%d) " FMT "\n",                  \
             blockIdx.x,                                \
             threadIdx.x, threadIdx.y);                 \
    } else {                                            \
      printf("(b%d t%d/%d/%d) " FMT "\n",               \
             blockIdx.x,                                \
             threadIdx.x, threadIdx.y, threadIdx.z);    \
    }                                                   \
  } else if (gridDim.z == 1) {                          \
    if (blockDim.y == 1 && blockDim.z == 1) {           \
      printf("(b%d/%d t%d) " FMT "\n",                  \
             blockIdx.x, blockIdx.y,                    \
             threadIdx.x);                              \
    } else if (blockDim.z == 1) {                       \
      printf("(b%d/%d t%d/%d) " FMT "\n",               \
             blockIdx.x, blockIdx.y,                    \
             threadIdx.x, threadIdx.y);                 \
    } else {                                            \
      printf("(b%d/%d t%d/%d/%d) " FMT "\n",            \
             blockIdx.x, blockIdx.y,                    \
             threadIdx.x, threadIdx.y, threadIdx.z);    \
    }                                                   \
  } else {                                              \
    if (blockDim.y == 1 && blockDim.z == 1) {           \
      printf("(b%d/%d/%d t%d) " FMT "\n",               \
             blockIdx.x, blockIdx.y, blockIdx.z,        \
             threadIdx.x);                              \
    } else if (blockDim.z == 1) {                       \
      printf("(b%d/%d/%d t%d/%d) " FMT "\n",            \
             blockIdx.x, blockIdx.y, blockIdx.z,        \
             threadIdx.x, threadIdx.y);                 \
    } else {                                            \
      printf("(b%d/%d/%d t%d/%d/%d) " FMT "\n",         \
             blockIdx.x, blockIdx.y, blockIdx.z,        \
             threadIdx.x, threadIdx.y, threadIdx.z);    \
    }                                                   \
  }                                                     \
}

//
// PRINT_UNIQUE_DEBUG
// Only prints once, on the first thread in the first block
//

__device__ __forceinline__
bool isFirstThread() {
  return (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
          blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0);
}

#define PRINT_UNIQUE_DEBUG(FMT, ...) {                                  \
    if (isFirstThread()) {                                              \
      PRINT_DEBUG(FMT, __VA_ARGS__);                                    \
    }                                                                   \
  }

#define PRINT_UNIQUE_DEBUG0(FMT) {                                      \
    if (isFirstThread()) {                                              \
      PRINT_DEBUG0(FMT);                                                \
    }                                                                   \
  }

#else

#define PRINT_DEBUG(FMT, ...)
#define PRINT_DEBUG0(FMT)

#define PRINT_UNIQUE_DEBUG(FMT, ...)
#define PRINT_UNIQUE_DEBUG0(FMT)

#endif // ENABLE_CUDA_DEBUG

} } // namespace
