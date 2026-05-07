#include "../include/matrix_mult.hpp"
#include <cmath>
#include <stdexcept>

std::vector<std::vector<double>> matrix_mult(
    const std::vector<std::vector<double>>& freq_mat,
    const std::vector<std::vector<double>>& prob_mat,
    const std::unordered_map<std::string, std::vector<double>>& params
) {
    if (freq_mat.empty() || prob_mat.empty()) {
        throw std::invalid_argument("Error: Input matrices cannot be empty.");
    }
    
    size_t n_rows = freq_mat.size();
    size_t n_cols = freq_mat[0].size();

    if (prob_mat.size() != n_rows || prob_mat[0].size() != n_cols) {
        throw std::invalid_argument("Error: Dimensions of freq_mat and prob_mat must strictly match.");
    }

    // 提取防止 log(0) 的极小容差 calc_tol
    double calc_tol = 1e-10;
    if (params.count("calc_tol") > 0 && !params.at("calc_tol").empty()) {
        calc_tol = params.at("calc_tol")[0];
    }

    std::vector<std::vector<double>> result(n_rows, std::vector<double>(n_cols, 0.0));

    // 针对每一行（在SDT中，同一类刺激的不同反应概率之和为1）进行矫正和计算
    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j < n_cols; ++j) {
            double p = prob_mat[i][j];
            double freq = freq_mat[i][j];

            // 1. 矫正概率，防止极小值导致的 log(0) 非法错误
            // 使用严谨的概率缩放公式，保证最小值不低于 tol 且单行总和完美维持为 1.0
            double p_adj = (1.0 - calc_tol * static_cast<double>(n_cols)) * p + calc_tol;

            // 2. 取对数，3. 与频数相乘
            result[i][j] = freq * std::log(p_adj);
        }
    }

    return result;
}