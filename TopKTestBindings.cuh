// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <vector>

namespace facebook { namespace cuda {

/// Binding for the unit test to test the warp-wide k-th element
/// selection code
float findTopKthElement(const std::vector<float>& data, int k);

/// Binding for the unit test to test the warp-wide top k element
/// selection code
std::vector<float> findTopKElements(const std::vector<float>& data, int k);

/// Binding for the unit test to test the warp-wide top k element +
/// index selection code, sorted by original index
std::vector<std::pair<float, int> >
findTopKElementsAndIndicesIndexOrder(const std::vector<float>& data, int k);

/// Binding for the unit test to test the warp-wide top k element +
/// index selection code, sorted by original value
std::vector<std::pair<float, int> >
findTopKElementsAndIndicesValueOrder(const std::vector<float>& data, int k);

} } // namespace
