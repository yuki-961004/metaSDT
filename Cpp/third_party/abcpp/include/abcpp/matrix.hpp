#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace abcpp {

class Matrix {
public:
    Matrix();

    Matrix(
        std::size_t rows,
        std::size_t cols,
        double value = 0.0
    );

    std::size_t rows() const;

    std::size_t cols() const;

    bool empty() const;

    double& operator()(std::size_t row, std::size_t col);

    double operator()(std::size_t row, std::size_t col) const;

    std::vector<double>& data();

    const std::vector<double>& data() const;

    std::vector<double> row(std::size_t index) const;

    std::vector<double> col(std::size_t index) const;

    void set_row(std::size_t index, const std::vector<double>& values);

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

Matrix from_row_major(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols
);

Matrix subset_rows(
    const Matrix& matrix,
    const std::vector<std::size_t>& rows
);

}  // namespace abcpp
