// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <vector>

namespace facebook { namespace cuda {

// Binding for the unit test to test the warp-wide sort code
std::vector<float> sort(const std::vector<float>& data);

// Binding for the unit test to test the warp-wide sort with indices code
std::vector<std::pair<float, int> >
sortWithIndices(const std::vector<float>& data);

} } // namespace
