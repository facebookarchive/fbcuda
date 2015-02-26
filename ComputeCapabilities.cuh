// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <cuda.h>

/** @file
    Compute capability specific defines that can be used as
    compile-time constants.
    warpSize for instance is not a compile-time constant, so it cannot
    be used for loop unrolling and register assignment.
*/

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 200
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define MAX_THREADS_PER_BLOCK 512
#elif __CUDA_ARCH__ <= 520
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define MAX_THREADS_PER_BLOCK 1024
#else
#error Unknown __CUDA_ARCH__; please define parameters
#endif // __CUDA_ARCH__ types
#endif // __CUDA_ARCH__

#ifndef __CUDA_ARCH__
// dummy value for host compiler
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define MAX_THREADS_PER_BLOCK 1024
#endif // !__CUDA_ARCH__

#define HALF_WARP_SIZE (WARP_SIZE / 2)
#define MAX_WARPS_PER_BLOCK (MAX_THREADS_PER_BLOCK / WARP_SIZE)
