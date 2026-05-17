#pragma once

#include "abcpp/result.hpp"

namespace abcpp {

SummaryResult summary(
    const AbcResult& result,
    bool unadjusted = false,
    double interval = 0.95
);

SummaryResult summarize(
    const AbcResult& result,
    bool unadjusted = false,
    double interval = 0.95
);

}  // namespace abcpp
