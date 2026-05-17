#pragma once

#include "abcpp/matrix.hpp"
#include "abcpp/options.hpp"
#include "abcpp/result.hpp"

#include <string>
#include <vector>

namespace abcpp {

AbcResult abc(
    const std::vector<double>& target,
    const Matrix& param,
    const Matrix& sumstat,
    const AbcOptions& options
);

AbcResult abc(
    const Matrix& target,
    const Matrix& param,
    const Matrix& sumstat,
    const AbcOptions& options
);

}  // namespace abcpp
