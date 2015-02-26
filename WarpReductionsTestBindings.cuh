// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <vector>

namespace facebook { namespace cuda {

// Binding for the unit test to test the warp-wide collision check
std::vector<int> hostCheckDuplicates(const std::vector<int>& v);

// Binding to return the colliding lane mask
unsigned int hostCheckDuplicateMask(const std::vector<int>& v);

} } // namespace
