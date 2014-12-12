// Copyright 2004-present Facebook. All Rights Reserved.
#include "cuda/SmallSortTestBindings.cuh"

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>
#include <unordered_set>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto ret = RUN_ALL_TESTS();

  cudaDeviceReset(); // to push stats cleanly
  return ret;
}

namespace facebook { namespace cuda {

namespace {

// Add in +/- inf, +/- 0, denorms
void addSpecialFloats(vector<float>& vals) {
  // Add in +/- infinity, with duplicates
  vals.push_back(numeric_limits<float>::infinity());
  vals.push_back(numeric_limits<float>::infinity());
  vals.push_back(-numeric_limits<float>::infinity());
  vals.push_back(-numeric_limits<float>::infinity());

  // Add in +/- zero, with duplicates
  vals.push_back(0.0f);
  vals.push_back(0.0f);
  vals.push_back(-0.0f);
  vals.push_back(-0.0f);

  // Add in some denorm floats, with duplicates
  vals.push_back(numeric_limits<float>::denorm_min() * 4.0f);
  vals.push_back(numeric_limits<float>::denorm_min());
  vals.push_back(numeric_limits<float>::denorm_min());
  vals.push_back(-numeric_limits<float>::denorm_min());
  vals.push_back(-numeric_limits<float>::denorm_min());
  vals.push_back(-numeric_limits<float>::denorm_min() * 4.0f);
}

} // namespace

TEST(SmallSort, weird) {
  vector<float> vals;
  addSpecialFloats(vals);

  vector<float> sorted = vals;
  sort(sorted.begin(), sorted.end(), std::greater<float>());

  for (int i = 0; i < 3; ++i) {
    shuffle(vals.begin(), vals.end(), random_device());
    auto out = sort(vals);

    ASSERT_EQ(sorted.size(), out.size());

    for (int j = 0; j < out.size(); ++j) {
      ASSERT_EQ(sorted[j], out[j]);
    }
  }
}

TEST(SmallSort, sortInRegisters) {
  // Test sorting vectors of size 1 to 4 x warpSize, which is the
  // maximum in-register size we support
  for (int size = 1; size <= 4 * 32; ++size) {
    vector<float> vals;

    for (int i = 0; i < size; ++i) {
      vals.push_back((float) i + 1);
    }

    vector<float> sorted = vals;
    sort(sorted.begin(), sorted.end(), std::greater<float>());

    for (int i = 0; i < 3; ++i) {
      shuffle(vals.begin(), vals.end(), random_device());
      auto out = sort(vals);

      ASSERT_EQ(sorted.size(), out.size());

      for (int j = 0; j < out.size(); ++j) {
        ASSERT_EQ(sorted[j], out[j]);
      }
    }
  }
}

TEST(SmallSort, sortIndicesInRegisters) {
  // Test sorting vectors of size 1 to 4 x warpSize, which is the
  // maximum in-register size we support
  for (int size = 1; size <= 4 * 32; ++size) {
    vector<float> vals;

    for (int i = 0; i < size; ++i) {
      vals.push_back((float) i);
    }

    vector<float> sorted = vals;
    sort(sorted.begin(), sorted.end(), std::greater<float>());

    for (int i = 0; i < 3; ++i) {
      shuffle(vals.begin(), vals.end(), random_device());
      auto out = sortWithIndices(vals);

      ASSERT_EQ(sorted.size(), out.size());

      for (int j = 0; j < out.size(); ++j) {
        ASSERT_EQ(sorted[j], out[j].first);

        int idx = out[j].second;
        ASSERT_GE(idx, 0);
        ASSERT_LT(idx, vals.size());
        ASSERT_EQ(out[j].first, vals[idx]);
      }

      // Test for uniqueness of indices
      unordered_set<int> indices;
      for (const auto p : out) {
        ASSERT_FALSE(indices.count(p.second));
        indices.emplace(p.second);
      }
    }
  }
}

} }
