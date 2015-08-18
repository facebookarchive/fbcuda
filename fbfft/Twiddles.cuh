// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "cuda/Complex.cuh"

namespace facebook { namespace cuda { namespace fbfft {

namespace {

#define PI 0x1.921FB6p+1f

/* Computed in Sollya as round(cos(k * pi / 32), single, RN)) */
#define FBFFT32_COSF_0  0x1.p0
#define FBFFT32_COSF_1  0xf.ec46dp-4
#define FBFFT32_COSF_2  0xf.b14bep-4
#define FBFFT32_COSF_3  0xf.4fa0bp-4
#define FBFFT32_COSF_4  0xe.c835ep-4
#define FBFFT32_COSF_5  0xe.1c598p-4
#define FBFFT32_COSF_6  0xd.4db31p-4
#define FBFFT32_COSF_7  0xc.5e403p-4
#define FBFFT32_COSF_8  0xb.504f3p-4
#define FBFFT32_COSF_9  0xa.26799p-4
#define FBFFT32_COSF_A  0x8.e39dap-4
#define FBFFT32_COSF_B  0x7.8ad75p-4
#define FBFFT32_COSF_C  0x6.1f78a8p-4
#define FBFFT32_COSF_D  0x4.a50188p-4
#define FBFFT32_COSF_E  0x3.1f1708p-4
#define FBFFT32_COSF_F  0x1.917a6cp-4
#define FBFFT32_COSF_G  0
#define FBFFT32_COSF_H  -0x1.917a6cp-4
#define FBFFT32_COSF_I  -0x3.1f1708p-4
#define FBFFT32_COSF_J  -0x4.a50188p-4
#define FBFFT32_COSF_K  -0x6.1f78a8p-4
#define FBFFT32_COSF_L  -0x7.8ad75p-4
#define FBFFT32_COSF_M  -0x8.e39dap-4
#define FBFFT32_COSF_N  -0xa.26799p-4
#define FBFFT32_COSF_O  -0xb.504f3p-4
#define FBFFT32_COSF_P  -0xc.5e403p-4
#define FBFFT32_COSF_Q  -0xd.4db31p-4
#define FBFFT32_COSF_R  -0xe.1c598p-4
#define FBFFT32_COSF_S  -0xe.c835ep-4
#define FBFFT32_COSF_T  -0xf.4fa0bp-4
#define FBFFT32_COSF_U  -0xf.b14bep-4
#define FBFFT32_COSF_V  -0xf.ec46dp-4

#define FBFFT32_SINF_0   0.0f
#define FBFFT32_SINF_1   FBFFT32_COSF_F
#define FBFFT32_SINF_2   FBFFT32_COSF_E
#define FBFFT32_SINF_3   FBFFT32_COSF_D
#define FBFFT32_SINF_4   FBFFT32_COSF_C
#define FBFFT32_SINF_5   FBFFT32_COSF_B
#define FBFFT32_SINF_6   FBFFT32_COSF_A
#define FBFFT32_SINF_7   FBFFT32_COSF_9
#define FBFFT32_SINF_8   FBFFT32_COSF_8
#define FBFFT32_SINF_9   FBFFT32_COSF_7
#define FBFFT32_SINF_A   FBFFT32_COSF_6
#define FBFFT32_SINF_B   FBFFT32_COSF_5
#define FBFFT32_SINF_C   FBFFT32_COSF_4
#define FBFFT32_SINF_D   FBFFT32_COSF_3
#define FBFFT32_SINF_E   FBFFT32_COSF_2
#define FBFFT32_SINF_F   FBFFT32_COSF_1
#define FBFFT32_SINF_G   1.0f
#define FBFFT32_SINF_H   FBFFT32_COSF_1
#define FBFFT32_SINF_I   FBFFT32_COSF_2
#define FBFFT32_SINF_J   FBFFT32_COSF_3
#define FBFFT32_SINF_K   FBFFT32_COSF_4
#define FBFFT32_SINF_L   FBFFT32_COSF_5
#define FBFFT32_SINF_M   FBFFT32_COSF_6
#define FBFFT32_SINF_N   FBFFT32_COSF_7
#define FBFFT32_SINF_O   FBFFT32_COSF_8
#define FBFFT32_SINF_P   FBFFT32_COSF_9
#define FBFFT32_SINF_Q   FBFFT32_COSF_A
#define FBFFT32_SINF_R   FBFFT32_COSF_B
#define FBFFT32_SINF_S   FBFFT32_COSF_C
#define FBFFT32_SINF_T   FBFFT32_COSF_D
#define FBFFT32_SINF_U   FBFFT32_COSF_E
#define FBFFT32_SINF_V   FBFFT32_COSF_F

#define FBFFT32_CEXPF_0 Complex(FBFFT32_COSF_0, FBFFT32_SINF_0)
#define FBFFT32_CEXPF_1 Complex(FBFFT32_COSF_1, FBFFT32_SINF_1)
#define FBFFT32_CEXPF_2 Complex(FBFFT32_COSF_2, FBFFT32_SINF_2)
#define FBFFT32_CEXPF_3 Complex(FBFFT32_COSF_3, FBFFT32_SINF_3)
#define FBFFT32_CEXPF_4 Complex(FBFFT32_COSF_4, FBFFT32_SINF_4)
#define FBFFT32_CEXPF_5 Complex(FBFFT32_COSF_5, FBFFT32_SINF_5)
#define FBFFT32_CEXPF_6 Complex(FBFFT32_COSF_6, FBFFT32_SINF_6)
#define FBFFT32_CEXPF_7 Complex(FBFFT32_COSF_7, FBFFT32_SINF_7)
#define FBFFT32_CEXPF_8 Complex(FBFFT32_COSF_8, FBFFT32_SINF_8)
#define FBFFT32_CEXPF_9 Complex(FBFFT32_COSF_9, FBFFT32_SINF_9)
#define FBFFT32_CEXPF_A Complex(FBFFT32_COSF_A, FBFFT32_SINF_A)
#define FBFFT32_CEXPF_B Complex(FBFFT32_COSF_B, FBFFT32_SINF_B)
#define FBFFT32_CEXPF_C Complex(FBFFT32_COSF_C, FBFFT32_SINF_C)
#define FBFFT32_CEXPF_D Complex(FBFFT32_COSF_D, FBFFT32_SINF_D)
#define FBFFT32_CEXPF_E Complex(FBFFT32_COSF_E, FBFFT32_SINF_E)
#define FBFFT32_CEXPF_F Complex(FBFFT32_COSF_F, FBFFT32_SINF_F)
#define FBFFT32_CEXPF_G Complex(FBFFT32_COSF_G, FBFFT32_SINF_G)
#define FBFFT32_CEXPF_H Complex(FBFFT32_COSF_H, FBFFT32_SINF_H)
#define FBFFT32_CEXPF_I Complex(FBFFT32_COSF_I, FBFFT32_SINF_I)
#define FBFFT32_CEXPF_J Complex(FBFFT32_COSF_J, FBFFT32_SINF_J)
#define FBFFT32_CEXPF_K Complex(FBFFT32_COSF_K, FBFFT32_SINF_K)
#define FBFFT32_CEXPF_L Complex(FBFFT32_COSF_L, FBFFT32_SINF_L)
#define FBFFT32_CEXPF_M Complex(FBFFT32_COSF_M, FBFFT32_SINF_M)
#define FBFFT32_CEXPF_N Complex(FBFFT32_COSF_N, FBFFT32_SINF_N)
#define FBFFT32_CEXPF_O Complex(FBFFT32_COSF_O, FBFFT32_SINF_O)
#define FBFFT32_CEXPF_P Complex(FBFFT32_COSF_P, FBFFT32_SINF_P)
#define FBFFT32_CEXPF_Q Complex(FBFFT32_COSF_Q, FBFFT32_SINF_Q)
#define FBFFT32_CEXPF_R Complex(FBFFT32_COSF_R, FBFFT32_SINF_R)
#define FBFFT32_CEXPF_S Complex(FBFFT32_COSF_S, FBFFT32_SINF_S)
#define FBFFT32_CEXPF_T Complex(FBFFT32_COSF_T, FBFFT32_SINF_T)
#define FBFFT32_CEXPF_U Complex(FBFFT32_COSF_U, FBFFT32_SINF_U)
#define FBFFT32_CEXPF_V Complex(FBFFT32_COSF_V, FBFFT32_SINF_V)

static const int kNumTwiddles = 128;

// __device__, __shared__ and __constant__ variables cannot be defined as
// external using the extern keyword. The only exception is for dynamically
// allocated __shared__ variables as described in Section B.2.3.
//
// Putting everything in an anon namespace, each .cu scope has its own
// array in global memory, and its own single time initialization.
//
// This is minor overhead.
//
float __device__ twiddleFactors[kNumTwiddles * 2];

/* Computed in Sollya as round(cos(k * pi / 128), single, RN)) */

const float twiddleFactorsHost[kNumTwiddles * 2] = {
  0x1.p0         , 0.0,
  0xf.fec43p-4   , 0x6.48558p-8,
  0xf.fb10fp-4   , 0xc.8fb3p-8,
  0xf.f4e6dp-4   , 0x1.2d520ap-4,
  0xf.ec46dp-4   , 0x1.917a6cp-4,
  0xf.e1324p-4   , 0x1.f564e6p-4,
  0xf.d3aacp-4   , 0x2.59020cp-4,
  0xf.c3b28p-4   , 0x2.bc4288p-4,
  0xf.b14bep-4   , 0x3.1f1708p-4,
  0xf.9c79dp-4   , 0x3.81704cp-4,
  0xf.853f8p-4   , 0x3.e33f3p-4,
  0xf.6ba07p-4   , 0x4.447498p-4,
  0xf.4fa0bp-4   , 0x4.a50188p-4,
  0xf.31447p-4   , 0x5.04d728p-4,
  0xf.10908p-4   , 0x5.63e6ap-4,
  0xe.ed89ep-4   , 0x5.c2215p-4,
  0xe.c835ep-4   , 0x6.1f78a8p-4,
  0xe.a09a7p-4   , 0x6.7bde5p-4,
  0xe.76bd8p-4   , 0x6.d744p-4,
  0xe.4aa59p-4   , 0x7.319ba8p-4,
  0xe.1c598p-4   , 0x7.8ad75p-4,
  0xd.ebe05p-4   , 0x7.e2e938p-4,
  0xd.b941ap-4   , 0x8.39c3dp-4,
  0xd.84853p-4   , 0x8.8f59bp-4,
  0xd.4db31p-4   , 0x8.e39dap-4,
  0xd.14d3dp-4   , 0x9.3682ap-4,
  0xc.d9f02p-4   , 0x9.87fcp-4,
  0xc.9d112p-4   , 0x9.d7fd1p-4,
  0xc.5e403p-4   , 0xa.26799p-4,
  0xc.1d87p-4    , 0xa.73656p-4,
  0xb.daef9p-4   , 0xa.beb4ap-4,
  0xb.96842p-4   , 0xb.085bbp-4,
  0xb.504f3p-4   , 0xb.504f3p-4,
  0xb.085bbp-4   , 0xb.96842p-4,
  0xa.beb4ap-4   , 0xb.daef9p-4,
  0xa.73656p-4   , 0xc.1d87p-4,
  0xa.26799p-4   , 0xc.5e403p-4,
  0x9.d7fd1p-4   , 0xc.9d112p-4,
  0x9.87fcp-4    , 0xc.d9f02p-4,
  0x9.3682ap-4   , 0xd.14d3dp-4,
  0x8.e39dap-4   , 0xd.4db31p-4,
  0x8.8f59bp-4   , 0xd.84853p-4,
  0x8.39c3dp-4   , 0xd.b941ap-4,
  0x7.e2e938p-4  , 0xd.ebe05p-4,
  0x7.8ad75p-4   , 0xe.1c598p-4,
  0x7.319ba8p-4  , 0xe.4aa59p-4,
  0x6.d744p-4    , 0xe.76bd8p-4,
  0x6.7bde5p-4   , 0xe.a09a7p-4,
  0x6.1f78a8p-4  , 0xe.c835ep-4,
  0x5.c2215p-4   , 0xe.ed89ep-4,
  0x5.63e6ap-4   , 0xf.10908p-4,
  0x5.04d728p-4  , 0xf.31447p-4,
  0x4.a50188p-4  , 0xf.4fa0bp-4,
  0x4.447498p-4  , 0xf.6ba07p-4,
  0x3.e33f3p-4   , 0xf.853f8p-4,
  0x3.81704cp-4  , 0xf.9c79dp-4,
  0x3.1f1708p-4  , 0xf.b14bep-4,
  0x2.bc4288p-4  , 0xf.c3b28p-4,
  0x2.59020cp-4  , 0xf.d3aacp-4,
  0x1.f564e6p-4  , 0xf.e1324p-4,
  0x1.917a6cp-4  , 0xf.ec46dp-4,
  0x1.2d520ap-4  , 0xf.f4e6dp-4,
  0xc.8fb3p-8    , 0xf.fb10fp-4,
  0x6.48558p-8   , 0xf.fec43p-4,
  0.0            , 0x1.p0,
  -0x6.48558p-8  , 0xf.fec43p-4,
  -0xc.8fb3p-8   , 0xf.fb10fp-4,
  -0x1.2d520ap-4 , 0xf.f4e6dp-4,
  -0x1.917a6cp-4 , 0xf.ec46dp-4,
  -0x1.f564e6p-4 , 0xf.e1324p-4,
  -0x2.59020cp-4 , 0xf.d3aacp-4,
  -0x2.bc4288p-4 , 0xf.c3b28p-4,
  -0x3.1f1708p-4 , 0xf.b14bep-4,
  -0x3.81704cp-4 , 0xf.9c79dp-4,
  -0x3.e33f3p-4  , 0xf.853f8p-4,
  -0x4.447498p-4 , 0xf.6ba07p-4,
  -0x4.a50188p-4 , 0xf.4fa0bp-4,
  -0x5.04d728p-4 , 0xf.31447p-4,
  -0x5.63e6ap-4  , 0xf.10908p-4,
  -0x5.c2215p-4  , 0xe.ed89ep-4,
  -0x6.1f78a8p-4 , 0xe.c835ep-4,
  -0x6.7bde5p-4  , 0xe.a09a7p-4,
  -0x6.d744p-4   , 0xe.76bd8p-4,
  -0x7.319ba8p-4 , 0xe.4aa59p-4,
  -0x7.8ad75p-4  , 0xe.1c598p-4,
  -0x7.e2e938p-4 , 0xd.ebe05p-4,
  -0x8.39c3dp-4  , 0xd.b941ap-4,
  -0x8.8f59bp-4  , 0xd.84853p-4,
  -0x8.e39dap-4  , 0xd.4db31p-4,
  -0x9.3682ap-4  , 0xd.14d3dp-4,
  -0x9.87fcp-4   , 0xc.d9f02p-4,
  -0x9.d7fd1p-4  , 0xc.9d112p-4,
  -0xa.26799p-4  , 0xc.5e403p-4,
  -0xa.73656p-4  , 0xc.1d87p-4,
  -0xa.beb4ap-4  , 0xb.daef9p-4,
  -0xb.085bbp-4  , 0xb.96842p-4,
  -0xb.504f3p-4  , 0xb.504f3p-4,
  -0xb.96842p-4  , 0xb.085bbp-4,
  -0xb.daef9p-4  , 0xa.beb4ap-4,
  -0xc.1d87p-4   , 0xa.73656p-4,
  -0xc.5e403p-4  , 0xa.26799p-4,
  -0xc.9d112p-4  , 0x9.d7fd1p-4,
  -0xc.d9f02p-4  , 0x9.87fcp-4,
  -0xd.14d3dp-4  , 0x9.3682ap-4,
  -0xd.4db31p-4  , 0x8.e39dap-4,
  -0xd.84853p-4  , 0x8.8f59bp-4,
  -0xd.b941ap-4  , 0x8.39c3dp-4,
  -0xd.ebe05p-4  , 0x7.e2e938p-4,
  -0xe.1c598p-4  , 0x7.8ad75p-4,
  -0xe.4aa59p-4  , 0x7.319ba8p-4,
  -0xe.76bd8p-4  , 0x6.d744p-4,
  -0xe.a09a7p-4  , 0x6.7bde5p-4,
  -0xe.c835ep-4  , 0x6.1f78a8p-4,
  -0xe.ed89ep-4  , 0x5.c2215p-4,
  -0xf.10908p-4  , 0x5.63e6ap-4,
  -0xf.31447p-4  , 0x5.04d728p-4,
  -0xf.4fa0bp-4  , 0x4.a50188p-4,
  -0xf.6ba07p-4  , 0x4.447498p-4,
  -0xf.853f8p-4  , 0x3.e33f3p-4,
  -0xf.9c79dp-4  , 0x3.81704cp-4,
  -0xf.b14bep-4  , 0x3.1f1708p-4,
  -0xf.c3b28p-4  , 0x2.bc4288p-4,
  -0xf.d3aacp-4  , 0x2.59020cp-4,
  -0xf.e1324p-4  , 0x1.f564e6p-4,
  -0xf.ec46dp-4  , 0x1.917a6cp-4,
  -0xf.f4e6dp-4  , 0x1.2d520ap-4,
  -0xf.fb10fp-4  , 0xc.8fb3p-8,
  -0xf.fec43p-4  , 0x6.48558p-8
};

void initTwiddles() {
  static bool firstTime = true;
  if (firstTime) {
    firstTime = false;
    cudaMemcpyToSymbol(twiddleFactors,
                       twiddleFactorsHost,
                       kNumTwiddles * sizeof(facebook::cuda::Complex));
  }
}

} // anon namespace

}}} // namespace
