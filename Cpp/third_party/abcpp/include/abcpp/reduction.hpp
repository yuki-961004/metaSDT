#pragma once

#include "abcpp/matrix.hpp"
#include "abcpp/options.hpp"
#include "abcpp/result.hpp"

#include <vector>

namespace abcpp {

struct ReducedSummary {
    Matrix sumstat;
    std::vector<double> target;
    ReductionInfo info;
};

ReducedSummary reduce_summary_statistics(
    const Matrix& param,
    const Matrix& sumstat,
    const std::vector<double>& target,
    const ReductionOptions& options
);

}  // namespace abcpp
