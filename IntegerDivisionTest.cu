// Copyright 2004-present Facebook. All Rights Reserved.
#include "cuda/IntegerDivision.cuh"

#include <gtest/gtest.h>
#include <stdlib.h>
#include <vector>

using namespace std;

namespace facebook { namespace cuda {

TEST(IntegerDivision, signed) {
  // Test divisors from 2 -> 500
  for (int d = 2; d <= 500; ++d) {
    FixedDivisor<int> sd(d);

    // Test random numerators > d
    int n = 1;
    for (int i = 0; i < 50; ++i) {
      while (true) {
        n = rand();
        if (n > d) {
          break;
        }
      }

      int magicQ = 0;
      int magicR = 0;
      sd.divMod(n, &magicQ, &magicR);

      int q = n / d;
      int r = n % d;
      ASSERT_EQ(q, magicQ);
      ASSERT_EQ(r, magicR);
    }


    // Test random numerators (all sizes)
    for (int i = 0; i < 50; ++i) {
      n = rand();

      int magicQ = 0;
      int magicR = 0;
      sd.divMod(n, &magicQ, &magicR);

      int q = n / d;
      int r = n % d;
      ASSERT_EQ(q, magicQ);
      ASSERT_EQ(r, magicR);
    }
  }
}

TEST(IntegerDivision, unsigned) {
  // Test divisors from 2 -> 500
  for (unsigned d = 2; d <= 500; ++d) {
    FixedDivisor<unsigned int> ud(d);

    // Test random numerators > d
    unsigned n = 1;
    for (int i = 0; i < 50; ++i) {
      while (true) {
        n = rand();
        if (n > d) {
          break;
        }
      }

      unsigned magicQ = 0;
      unsigned magicR = 0;
      ud.divMod(n, &magicQ, &magicR);

      unsigned q = n / d;
      unsigned r = n % d;
      ASSERT_EQ(q, magicQ);
      ASSERT_EQ(r, magicR);
    }


    // Test random numerators (all sizes)
    for (int i = 0; i < 50; ++i) {
      n = rand();

      unsigned magicQ = 0;
      unsigned magicR = 0;
      ud.divMod(n, &magicQ, &magicR);

      unsigned q = n / d;
      unsigned r = n % d;
      ASSERT_EQ(q, magicQ);
      ASSERT_EQ(r, magicR);
    }
  }
}

const int kNumTests = 1000;

__global__ void signedGPU(int numTests,
                          int* ns,
                          int* ds,
                          int* qs,
                          int* rs) {
  for (int i = 0; i < numTests; ++i) {
    int n = ns[i];
    int d = ds[i];

    qs[i] = n / d;
    rs[i] = n % d;
  }
}

TEST(IntegerDivision, signedGPU) {
  vector<int> random;
  vector<int> randomDiv;
  vector<int> randomQ;
  vector<int> randomR;

  for (int i = 0; i < kNumTests; ++i) {
    int val = rand();
    random.push_back(val);

    int d = 0;
    while (true) {
      d = rand();
      if (d >= 2) {
        break;
      }
    }

    randomDiv.push_back(d);

    randomQ.push_back(val / d);
    randomR.push_back(val % d);
  }

  int* devRandom = NULL;
  cudaMalloc(&devRandom, kNumTests * sizeof(int));
  cudaMemcpy(devRandom, &random[0],
             kNumTests * sizeof(int), cudaMemcpyHostToDevice);

  int* devRandomDiv = NULL;
  cudaMalloc(&devRandomDiv, kNumTests * sizeof(int));
  cudaMemcpy(devRandomDiv, &randomDiv[0],
             kNumTests * sizeof(int), cudaMemcpyHostToDevice);

  int* devRandomQ = NULL;
  cudaMalloc(&devRandomQ, kNumTests * sizeof(int));
  int* devRandomR = NULL;
  cudaMalloc(&devRandomR, kNumTests * sizeof(int));

  signedGPU<<<1, 1>>>(
    kNumTests, devRandom, devRandomDiv, devRandomQ, devRandomR);

  vector<int> gpuQ(kNumTests);
  cudaMemcpy(&gpuQ[0], devRandomQ,
             kNumTests * sizeof(int), cudaMemcpyDeviceToHost);
  vector<int> gpuR(kNumTests);
  cudaMemcpy(&gpuR[0], devRandomR,
             kNumTests * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < kNumTests; ++i) {
    ASSERT_EQ(randomQ[i], gpuQ[i]);
    ASSERT_EQ(randomR[i], gpuR[i]);
  }

  cudaFree(devRandom);
  cudaFree(devRandomDiv);
  cudaFree(devRandomQ);
  cudaFree(devRandomR);
}

__global__ void unsignedGPU(int numTests,
                            unsigned* ns,
                            unsigned* ds,
                            unsigned* qs,
                            unsigned* rs) {
  for (int i = 0; i < numTests; ++i) {
    unsigned n = ns[i];
    unsigned d = ds[i];

    qs[i] = n / d;
    rs[i] = n % d;
  }
}

TEST(IntegerDivision, unsignedGPU) {
  vector<unsigned> random;
  vector<unsigned> randomDiv;
  vector<unsigned> randomQ;
  vector<unsigned> randomR;

  for (int i = 0; i < kNumTests; ++i) {
    unsigned val = rand();
    random.push_back(val);

    unsigned d = 0;
    while (true) {
      d = rand();
      if (d >= 2) {
        break;
      }
    }

    randomDiv.push_back(d);

    randomQ.push_back(val / d);
    randomR.push_back(val % d);
  }

  unsigned* devRandom = NULL;
  cudaMalloc(&devRandom, kNumTests * sizeof(unsigned));
  cudaMemcpy(devRandom, &random[0],
             kNumTests * sizeof(unsigned), cudaMemcpyHostToDevice);

  unsigned* devRandomDiv = NULL;
  cudaMalloc(&devRandomDiv, kNumTests * sizeof(unsigned));
  cudaMemcpy(devRandomDiv, &randomDiv[0],
             kNumTests * sizeof(unsigned), cudaMemcpyHostToDevice);

  unsigned* devRandomQ = NULL;
  cudaMalloc(&devRandomQ, kNumTests * sizeof(unsigned));
  unsigned* devRandomR = NULL;
  cudaMalloc(&devRandomR, kNumTests * sizeof(unsigned));

  unsignedGPU<<<1, 1>>>(
    kNumTests, devRandom, devRandomDiv, devRandomQ, devRandomR);

  vector<unsigned> gpuQ(kNumTests);
  cudaMemcpy(&gpuQ[0], devRandomQ,
             kNumTests * sizeof(unsigned), cudaMemcpyDeviceToHost);
  vector<unsigned> gpuR(kNumTests);
  cudaMemcpy(&gpuR[0], devRandomR,
             kNumTests * sizeof(unsigned), cudaMemcpyDeviceToHost);

  for (int i = 0; i < kNumTests; ++i) {
    ASSERT_EQ(randomQ[i], gpuQ[i]);
    ASSERT_EQ(randomR[i], gpuR[i]);
  }

  cudaFree(devRandom);
  cudaFree(devRandomDiv);
  cudaFree(devRandomQ);
  cudaFree(devRandomR);
}

} }
