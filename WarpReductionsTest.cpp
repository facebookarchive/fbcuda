// Copyright 2004-present Facebook. All Rights Reserved.
#include "cuda/WarpReductionsTestBindings.cuh"

#include <gtest/gtest.h>
#include <iterator>
#include <stdlib.h>
#include <vector>

using namespace std;

namespace facebook { namespace cuda {

TEST(WarpReductions, collision) {
  for (int numDups = 0; numDups < 32; ++numDups) {

    vector<int> v;
    for (int i = 0; i < 32 - numDups; ++i) {
      int r = 0;

      while (true) {
        r = rand();

        // C++11 std::find doesn't work with nvcc now
        bool found = false;
        for (int i = 0; i < v.size(); ++i) {
          if (v[i] == r) {
            found = true;
            break;
          }
        }

        if (!found) {
          break;
        }
      }

      v.push_back(r);
    }

    for (int i = 0; i < numDups; ++i) {
      v.push_back(v[0]);
    }

    EXPECT_EQ(32, v.size());
    auto dupCheck = hostCheckDuplicates(v);

    for (auto dup : dupCheck) {
      ASSERT_EQ((numDups > 0), dup);
    }
  }
}

TEST(WarpReductions, collisionMask) {
  for (int numDups = 0; numDups < 32; ++numDups) {
    vector<int> v;
    for (int i = 0; i < 32 - numDups; ++i) {
      int r = 0;

      while (true) {
        r = rand();

        // C++11 std::find doesn't work with nvcc now
        bool found = false;
        for (int i = 0; i < v.size(); ++i) {
          if (v[i] == r) {
            found = true;
            break;
          }
        }

        if (!found) {
          break;
        }
      }

      v.push_back(r);
    }

    for (int i = 0; i < numDups; ++i) {
      v.push_back(v[0]);
    }

    EXPECT_EQ(32, v.size());

    auto mask = hostCheckDuplicateMask(v);
    auto expected = numDups > 0 ? 0xffffffffU << (32 - numDups) : 0;
    ASSERT_EQ(expected, mask);
  }
}

} }
