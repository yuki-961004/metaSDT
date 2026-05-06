#ifndef MATRIX_PROB_HPP
#define MATRIX_PROB_HPP

#include <vector>
#include <string>
#include <unordered_map>

// 概率矩阵结构体，其结构与 matrix_freq 返回的频数矩阵完美对应
struct MatrixProb {
    std::vector<std::vector<double>> prob_mat;
    std::vector<std::string> row_names;
    std::vector<std::string> col_names;
};

MatrixProb matrix_prob(
    const std::vector<double>& cdf_noise,
    const std::vector<double>& cdf_signal,
    const std::unordered_map<std::string, std::vector<double>>& params
);

#endif // MATRIX_PROB_HPP