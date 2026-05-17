#pragma once

#include "abcpp/matrix.hpp"

#include <cstddef>
#include <vector>

namespace abcpp {

bool is_finite(double value);

double median(std::vector<double> values);

double quantile_type7(std::vector<double> values, double probability);

double mad(const std::vector<double>& values);

double mean(const std::vector<double>& values);

double weighted_mean(
    const std::vector<double>& values,
    const std::vector<double>& weights
);

double sample_sd(const std::vector<double>& values);

std::vector<double> column_means(const Matrix& matrix);

std::vector<double> column_mads(
    const Matrix& matrix,
    const std::vector<bool>& keep_rows
);

std::vector<double> column_weighted_sse(
    const Matrix& matrix,
    const std::vector<double>& weights
);

double kernel_mode(
    const std::vector<double>& values,
    const std::vector<double>& weights
);

}  // namespace abcpp
