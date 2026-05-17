#pragma once

#include "abcpp/matrix.hpp"

#include <vector>

namespace abcpp {

Matrix transpose(const Matrix& matrix);

Matrix multiply(const Matrix& left, const Matrix& right);

std::vector<double> multiply(
    const Matrix& matrix,
    const std::vector<double>& vector
);

Matrix add_intercept(const Matrix& x);

std::vector<double> solve_linear_system(
    Matrix a,
    std::vector<double> b
);

Matrix inverse(Matrix matrix);

Matrix weighted_least_squares(
    const Matrix& design,
    const Matrix& response,
    const std::vector<double>& weights,
    double ridge_lambda
);

Matrix predict(const Matrix& design, const Matrix& coefficients);

}  // namespace abcpp
