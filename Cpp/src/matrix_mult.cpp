#include "../include/matrix_mult.hpp"
#include <cmath>
#include <stdexcept>

template <typename T>
std::vector<std::vector<T>> matrix_mult(
    const std::vector<std::vector<double>>& freq_mat,
    const std::vector<std::vector<T>>& prob_mat,
    const std::unordered_map<std::string, std::vector<T>>& std_params
) {
    // ==========================================================
    // 1. 安全检查与维度校验
    // ==========================================================
    if (freq_mat.empty() || prob_mat.empty()) {
        throw std::invalid_argument("Error: Input matrices cannot be empty.");
    }
    
    size_t n_rows = freq_mat.size();
    size_t n_cols = freq_mat[0].size();

    if (prob_mat.size() != n_rows || prob_mat[0].size() != n_cols) {
        throw std::invalid_argument(
            "Error: Dimensions of freq_mat and prob_mat must strictly match."
        );
    }

    // ==========================================================
    // 2. 提取全局极小容差参数
    // ==========================================================
    // 提取防止 log(0) 的极小容差 calc_tol
    double calc_tol = 1e-10;
    if (std_params.count("calc_tol") > 0 && !std_params.at("calc_tol").empty()) {
        calc_tol = static_cast<double>(std_params.at("calc_tol")[0]);
    }

    std::vector<std::vector<T>> result(n_rows, std::vector<T>(n_cols, 0.0));

    // ==========================================================
    // 3. 核心矩阵运算与修正 (Freq * log(Prob))
    // ==========================================================
    // 针对每一行（在SDT中，同一类刺激的不同反应概率之和为1）进行矫正和计算
    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j < n_cols; ++j) {
            T p = prob_mat[i][j];
            double freq = freq_mat[i][j];

            // 1. 矫正概率，防止极小值导致的 log(0) 非法错误
            // 保证最小值不低于 tol 且单行总和完美维持为 1.0
            T p_adj = (1.0 - calc_tol * static_cast<double>(n_cols)) * 
                           p + calc_tol;

            // 2. 取对数 (ADL Hygiene)，3. 与频数相乘
            using std::log;
            result[i][j] = freq * log(p_adj);
        }
    }

    return result;
}

// 显式实例化 double 版本以兼容现有代码
template std::vector<std::vector<double>> matrix_mult<double>(
    const std::vector<std::vector<double>>&, 
    const std::vector<std::vector<double>>&, 
    const std::unordered_map<std::string, std::vector<double>>&);