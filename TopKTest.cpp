// Copyright 2004-present Facebook. All Rights Reserved.
#include "cuda/TopKTestBindings.cuh"

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


struct ValueOrder {
  bool operator()(const pair<float, int>& lhs,
                  const pair<float, int>& rhs) {
    return (lhs.first > rhs.first) ||
      ((lhs.first == rhs.first) && (lhs.second > rhs.second));
  }
};

void
testSorting(const vector<float>& shuffled,
            const vector<float>& sorted,
            int k) {
  auto outIndex = findTopKElementsAndIndicesIndexOrder(shuffled, k);

  // Verify that outIndex is in index order, and are unique
  for (int j = 0; j < outIndex.size() - 1; ++j) {
    ASSERT_LT(outIndex[j].second, outIndex[j + 1].second);
  }

  // Test that outIndex indexes the same element
  for (int j = 0; j < outIndex.size(); ++j) {
    int idx = outIndex[j].second;

    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, shuffled.size());
    ASSERT_EQ(outIndex[j].first, shuffled[idx]);
  }

  // Sorting outIndex by value order should result in something
  // comparable with `sorted`
  sort(outIndex.begin(), outIndex.end(), ValueOrder());

  for (int j = 0; j < outIndex.size(); ++j) {
    ASSERT_EQ(sorted[j], outIndex[j].first);

    int idx = outIndex[j].second;
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, shuffled.size());
    ASSERT_EQ(outIndex[j].first, shuffled[idx]);
  }

  if (k <= 128) {
    // Sorting by value is implemented too; test that
    auto outValue = findTopKElementsAndIndicesValueOrder(shuffled, k);
    ASSERT_EQ(outValue.size(), outIndex.size());

    // Test sorted index
    for (int j = 0; j < outValue.size(); ++j) {
      ASSERT_EQ(sorted[j], outValue[j].first);

      int idx = outValue[j].second;
      ASSERT_GE(idx, 0);
      ASSERT_LT(idx, shuffled.size());
      ASSERT_EQ(outValue[j].first, shuffled[idx]);
    }

    // Test for uniqueness of indices
    unordered_set<int> indices;
    for (const auto p : outValue) {
      ASSERT_FALSE(indices.count(p.second));
      indices.emplace(p.second);
    }
  }
}

} // namespace

TEST(TopK, basicSmall) {
  vector<float> vals = {2.0f, 3.0f, 5.0f, 11.0f};
  shuffle(vals.begin(), vals.end(), random_device());

  EXPECT_EQ(11.0f, findTopKthElement(vals, 1));
  EXPECT_EQ(5.0f, findTopKthElement(vals, 2));
  EXPECT_EQ(3.0f, findTopKthElement(vals, 3));
  EXPECT_EQ(2.0f, findTopKthElement(vals, 4));
}

TEST(TopK, basicLarge) {
  vector<float> vals;
  for (int i = 0; i <= 50; ++i) {
    vals.push_back((float) i * 2);
  }

  shuffle(vals.begin(), vals.end(), random_device());

  for (int i = 0; i <= 50; ++i) {
    EXPECT_EQ((50 - i) * 2,
              findTopKthElement(vals, i + 1));
  }
}

TEST(TopK, negativeSmall) {
  vector<float> vals = {1.0f, -2.0f, -3.0f, -5.0f, -11.0f};
  shuffle(vals.begin(), vals.end(), random_device());

  EXPECT_EQ(1.0f, findTopKthElement(vals, 1));
  EXPECT_EQ(-2.0f, findTopKthElement(vals, 2));
  EXPECT_EQ(-3.0f, findTopKthElement(vals, 3));
  EXPECT_EQ(-5.0f, findTopKthElement(vals, 4));
  EXPECT_EQ(-11.0f, findTopKthElement(vals, 5));
}

TEST(TopK, weird) {
  // As long as it is not a NaN, we should deal with it properly.
  vector<float> vals = {
    numeric_limits<float>::infinity(),
    0.0f,
    numeric_limits<float>::denorm_min(),
    -0.0f,
    -numeric_limits<float>::infinity(),
    -numeric_limits<float>::denorm_min(),
    4.0f * numeric_limits<float>::denorm_min()
  };

  shuffle(vals.begin(), vals.end(), random_device());

  EXPECT_EQ(numeric_limits<float>::infinity(),
            findTopKthElement(vals, 1));
  EXPECT_EQ(4.0f * numeric_limits<float>::denorm_min(),
            findTopKthElement(vals, 2));
  EXPECT_EQ(numeric_limits<float>::denorm_min(),
            findTopKthElement(vals, 3));
  EXPECT_EQ(0.0f,
            findTopKthElement(vals, 4));
  EXPECT_EQ(-0.0f,
            findTopKthElement(vals, 5));
  EXPECT_EQ(-numeric_limits<float>::denorm_min(),
            findTopKthElement(vals, 6));
  EXPECT_EQ(-numeric_limits<float>::infinity(),
            findTopKthElement(vals, 7));
}

TEST(TopK, nonUnique) {
  vector<float> vals;
  for (int i = 0; i < 16; ++i) {
    vals.push_back(1.0f);
  }
  for (int i = 16; i < 32; ++i) {
    vals.push_back(2.0f);
  }

  EXPECT_EQ(2.0f, findTopKthElement(vals, 1));
  EXPECT_EQ(2.0f, findTopKthElement(vals, 2));
  EXPECT_EQ(2.0f, findTopKthElement(vals, 16));
  EXPECT_EQ(1.0f, findTopKthElement(vals, 17));
  EXPECT_EQ(1.0f, findTopKthElement(vals, 32));
}

TEST(TopK, random) {
  random_device dev;
  mt19937 gen(dev());
  normal_distribution<float> dist(0, 1e6f);
  uniform_real_distribution<float> smallDist(-1e-10f, 1e-10f);

  vector<float> vals;
  int dupsRemaining = 50;
  for (int i = 0; i < 1000; ++i) {
    const auto val = dist(gen);
    vals.push_back(val);

    const auto smallVal = smallDist(gen);
    vals.push_back(smallVal);

    // Also add in some duplicate entries
    if (dupsRemaining-- > 0) {
      vals.push_back(val);
      vals.push_back(smallVal);
    }
  }

  addSpecialFloats(vals);

  auto sorted = vals;
  sort(sorted.begin(), sorted.end(), std::greater<float>());

  for (int i = 0; i < vals.size(); ++i) {
    ASSERT_EQ(sorted[i], findTopKthElement(vals, i + 1));
  }
}

TEST(TopK, topKSortedWeird) {
  // As long as it is not a NaN, we should deal with it properly.
  vector<float> vals;
  addSpecialFloats(vals);

  auto sorted = vals;
  sort(sorted.begin(), sorted.end(), std::greater<float>());

  for (int i = 0; i < vals.size(); ++i) {
    shuffle(vals.begin(), vals.end(), random_device());
    auto out = findTopKElements(vals, i + 1);

    for (int j = 0; j < out.size(); ++j) {
      ASSERT_EQ(sorted[j], out[j]);
    }
  }
}

TEST(TopK, topKSortedSmall) {
  vector<float> vals;

  // A vector smaller than the warp size
  for (int i = 0; i <= 10; ++i) {
    vals.push_back((float) i);
  }

  // Add one duplicate value too
  vals.push_back(5.0f);

  auto sorted = vals;
  sort(sorted.begin(), sorted.end(), std::greater<float>());

  for (int i = 0; i < vals.size(); ++i) {
    shuffle(vals.begin(), vals.end(), random_device());
    auto out = findTopKElements(vals, i + 1);

    for (int j = 0; j < out.size(); ++j) {
      ASSERT_EQ(sorted[j], out[j]);
    }
  }
}

TEST(TopK, topKSortedInWarp) {
  for (int len = 0; len < 3 * 32; ++len) {
    vector<float> vals;

    for (int i = 0; i < len; ++i) {
      vals.push_back((float) i);
    }

    auto sorted = vals;
    sort(sorted.begin(), sorted.end(), std::greater<float>());

    // Test top-k for all sizes up to `len`
    for (int k = 1; k <= len; ++k) {
      shuffle(vals.begin(), vals.end(), random_device());
      auto out = findTopKElements(vals, k);

      for (int j = 0; j < out.size(); ++j) {
        ASSERT_EQ(sorted[j], out[j]);
      }
    }
  }
}

TEST(TopK, topKSortedLarge) {
  vector<float> vals;

  // A vector larger than the warp size
  for (int i = 0; i <= 1000; ++i) {
    vals.push_back((float) i);
  }

  // Add a top duplicated element too
  vals.push_back(998.0f);

  auto sorted = vals;
  sort(sorted.begin(), sorted.end(), std::greater<float>());

  // Algorithm only deals with k <= 32 at the moment
  for (int i = 0; i < 32; ++i) {
    shuffle(vals.begin(), vals.end(), random_device());
    auto out = findTopKElements(vals, i + 1);

    for (int j = 0; j < out.size(); ++j) {
      ASSERT_EQ(sorted[j], out[j]);
    }
  }
}

TEST(TopK, sortedIndicesSmall) {
  vector<float> vals;

  for (int i = 0; i <= 10; ++i) {
    vals.push_back((float) i);
  }

  // Add one duplicate normal value too
  vals.push_back(5.0f);
  addSpecialFloats(vals);

  auto sorted = vals;
  sort(sorted.begin(), sorted.end(), std::greater<float>());

  // Test top-k for all sizes
  for (int k = 1; k <= vals.size(); ++k) {
    shuffle(vals.begin(), vals.end(), random_device());
    testSorting(vals, sorted, k);
  }
}

TEST(TopK, sortedIndicesInWarp) {
  for (int len = 1; len <= 3 * 32; ++len) {
    vector<float> vals;

    for (int i = 0; i < len; ++i) {
      vals.push_back((float) i);
    }

    auto sorted = vals;
    sort(sorted.begin(), sorted.end(), std::greater<float>());

    // Test top-k for all sizes
    for (int k = 1; k <= vals.size(); ++k) {
      shuffle(vals.begin(), vals.end(), random_device());
      testSorting(vals, sorted, k);
    }
  }
}

TEST(TopK, sortedIndicesLarge) {
  random_device dev;
  mt19937 gen(dev());
  normal_distribution<float> dist(0, 1e6f);

  vector<float> vals;
  int dupsRemaining = 200;

  // A vector larger than the warp size
  for (int i = 0; i <= 500; ++i) {
    const auto val = dist(gen);
    vals.push_back(val);

    if (dupsRemaining-- > 0) {
      vals.push_back(val);
    }
  }

  addSpecialFloats(vals);

  auto sorted = vals;
  sort(sorted.begin(), sorted.end(), std::greater<float>());

  // Test top-k for all sizes
  for (int k = 1; k <= vals.size(); ++k) {
    shuffle(vals.begin(), vals.end(), random_device());
    testSorting(vals, sorted, k);
  }
}

} }
