#include "abcpp/matrix.hpp"

#include <string>

namespace abcpp {

Matrix::Matrix() = default;

Matrix::Matrix(
    std::size_t rows,
    std::size_t cols,
    double value
) : rows_(rows),
    cols_(cols),
    data_(rows * cols, value) {}

std::size_t Matrix::rows() const {
    return rows_;
}

std::size_t Matrix::cols() const {
    return cols_;
}

bool Matrix::empty() const {
    return rows_ == 0 || cols_ == 0;
}

double& Matrix::operator()(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index is out of range.");
    }
    return data_[row * cols_ + col];
}

double Matrix::operator()(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index is out of range.");
    }
    return data_[row * cols_ + col];
}

std::vector<double>& Matrix::data() {
    return data_;
}

const std::vector<double>& Matrix::data() const {
    return data_;
}

std::vector<double> Matrix::row(std::size_t index) const {
    if (index >= rows_) {
        throw std::out_of_range("Matrix row index is out of range.");
    }

    std::vector<double> out(cols_, 0.0);
    for (std::size_t col = 0; col < cols_; ++col) {
        out[col] = (*this)(index, col);
    }
    return out;
}

std::vector<double> Matrix::col(std::size_t index) const {
    if (index >= cols_) {
        throw std::out_of_range("Matrix column index is out of range.");
    }

    std::vector<double> out(rows_, 0.0);
    for (std::size_t row = 0; row < rows_; ++row) {
        out[row] = (*this)(row, index);
    }
    return out;
}

void Matrix::set_row(
    std::size_t index,
    const std::vector<double>& values
) {
    if (index >= rows_) {
        throw std::out_of_range("Matrix row index is out of range.");
    }
    if (values.size() != cols_) {
        throw std::invalid_argument("Matrix row has incompatible length.");
    }

    for (std::size_t col = 0; col < cols_; ++col) {
        (*this)(index, col) = values[col];
    }
}

Matrix from_row_major(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols
) {
    if (values.size() != rows * cols) {
        throw std::invalid_argument("Row-major data has wrong length.");
    }

    Matrix out(rows, cols);
    out.data() = values;
    return out;
}

Matrix subset_rows(
    const Matrix& matrix,
    const std::vector<std::size_t>& rows
) {
    Matrix out(rows.size(), matrix.cols());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            out(i, col) = matrix(rows[i], col);
        }
    }
    return out;
}

}  // namespace abcpp
