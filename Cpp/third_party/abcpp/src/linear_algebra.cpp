#include "abcpp/linear_algebra.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace abcpp {

Matrix transpose(const Matrix& matrix) {
    Matrix out(matrix.cols(), matrix.rows());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            out(col, row) = matrix(row, col);
        }
    }
    return out;
}

Matrix multiply(const Matrix& left, const Matrix& right) {
    if (left.cols() != right.rows()) {
        throw std::invalid_argument("Matrix product has incompatible shape.");
    }

    Matrix out(left.rows(), right.cols(), 0.0);
    for (std::size_t row = 0; row < left.rows(); ++row) {
        for (std::size_t mid = 0; mid < left.cols(); ++mid) {
            const double left_value = left(row, mid);
            for (std::size_t col = 0; col < right.cols(); ++col) {
                out(row, col) += left_value * right(mid, col);
            }
        }
    }
    return out;
}

std::vector<double> multiply(
    const Matrix& matrix,
    const std::vector<double>& vector
) {
    if (matrix.cols() != vector.size()) {
        throw std::invalid_argument("Matrix-vector product has wrong shape.");
    }

    std::vector<double> out(matrix.rows(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            out[row] += matrix(row, col) * vector[col];
        }
    }
    return out;
}

Matrix add_intercept(const Matrix& x) {
    Matrix out(x.rows(), x.cols() + 1, 1.0);
    for (std::size_t row = 0; row < x.rows(); ++row) {
        for (std::size_t col = 0; col < x.cols(); ++col) {
            out(row, col + 1) = x(row, col);
        }
    }
    return out;
}

std::vector<double> solve_linear_system(
    Matrix a,
    std::vector<double> b
) {
    const std::size_t n = a.rows();
    if (a.rows() != a.cols() || b.size() != n) {
        throw std::invalid_argument("Linear system has incompatible shape.");
    }

    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t best = pivot;
        double best_abs = std::fabs(a(pivot, pivot));
        for (std::size_t row = pivot + 1; row < n; ++row) {
            const double candidate = std::fabs(a(row, pivot));
            if (candidate > best_abs) {
                best = row;
                best_abs = candidate;
            }
        }

        if (best_abs < 1e-12) {
            a(pivot, pivot) += 1e-8;
            best_abs = std::fabs(a(pivot, pivot));
        }
        if (best_abs < 1e-14) {
            throw std::runtime_error("Linear system is singular.");
        }

        if (best != pivot) {
            for (std::size_t col = pivot; col < n; ++col) {
                std::swap(a(pivot, col), a(best, col));
            }
            std::swap(b[pivot], b[best]);
        }

        const double diag = a(pivot, pivot);
        for (std::size_t col = pivot; col < n; ++col) {
            a(pivot, col) /= diag;
        }
        b[pivot] /= diag;

        for (std::size_t row = 0; row < n; ++row) {
            if (row == pivot) {
                continue;
            }
            const double factor = a(row, pivot);
            for (std::size_t col = pivot; col < n; ++col) {
                a(row, col) -= factor * a(pivot, col);
            }
            b[row] -= factor * b[pivot];
        }
    }

    return b;
}

Matrix inverse(Matrix matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Only square matrices can be inverted.");
    }

    Matrix out(matrix.rows(), matrix.cols(), 0.0);
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
        std::vector<double> basis(matrix.rows(), 0.0);
        basis[col] = 1.0;
        const std::vector<double> solved = solve_linear_system(
            matrix,
            basis
        );
        for (std::size_t row = 0; row < matrix.rows(); ++row) {
            out(row, col) = solved[row];
        }
    }
    return out;
}

Matrix weighted_least_squares(
    const Matrix& design,
    const Matrix& response,
    const std::vector<double>& weights,
    double ridge_lambda
) {
    if (design.rows() != response.rows() || design.rows() != weights.size()) {
        throw std::invalid_argument("Regression inputs have wrong shape.");
    }

    Matrix xtwx(design.cols(), design.cols(), 0.0);
    Matrix xtwy(design.cols(), response.cols(), 0.0);

    for (std::size_t row = 0; row < design.rows(); ++row) {
        const double weight = std::max(weights[row], 0.0);
        for (std::size_t left = 0; left < design.cols(); ++left) {
            for (std::size_t right = 0; right < design.cols(); ++right) {
                xtwx(left, right) +=
                    weight * design(row, left) * design(row, right);
            }
            for (std::size_t col = 0; col < response.cols(); ++col) {
                xtwy(left, col) +=
                    weight * design(row, left) * response(row, col);
            }
        }
    }

    for (std::size_t diag = 1; diag < design.cols(); ++diag) {
        xtwx(diag, diag) += ridge_lambda;
    }

    for (std::size_t diag = 0; diag < design.cols(); ++diag) {
        xtwx(diag, diag) += 1e-10;
    }

    Matrix coefficients(design.cols(), response.cols(), 0.0);
    for (std::size_t col = 0; col < response.cols(); ++col) {
        std::vector<double> rhs(design.cols(), 0.0);
        for (std::size_t row = 0; row < design.cols(); ++row) {
            rhs[row] = xtwy(row, col);
        }
        const std::vector<double> solved = solve_linear_system(xtwx, rhs);
        for (std::size_t row = 0; row < design.cols(); ++row) {
            coefficients(row, col) = solved[row];
        }
    }

    return coefficients;
}

Matrix predict(const Matrix& design, const Matrix& coefficients) {
    return multiply(design, coefficients);
}

}  // namespace abcpp
