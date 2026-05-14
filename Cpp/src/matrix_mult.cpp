#include "../include/matrix_mult.hpp"
#include <cmath>
#include <stdexcept>

/* ========================================================================== *
 *                    Frequency-Probability Matrix Module                     *
 * ========================================================================== */

template <typename T>
std::vector<std::vector<std::vector<T>>> matrix_mult(
    const std::vector<std::vector<std::vector<double>>>& freq_mat,
    const std::vector<std::vector<std::vector<T>>>& prob_mat,
    const std::unordered_map<std::string, std::vector<T>>& std_params
) {
/* ========================================================================== *
 *                   1. Input Validation and Dimensions                       *
 * ========================================================================== */
    if (freq_mat.empty() || prob_mat.empty()) {
        throw std::invalid_argument("Error: Input matrices cannot be empty.");
    }

    size_t n_diffs = freq_mat.size();
    size_t n_rows = freq_mat[0].size();
    size_t n_cols = freq_mat[0][0].size();

    if (
        prob_mat.size() != n_diffs ||
        prob_mat[0].size() != n_rows ||
        prob_mat[0][0].size() != n_cols
    ) {
        throw std::invalid_argument(
            "Error: Dimension mismatch! freq_mat is [" +
            std::to_string(n_diffs) +
            " x " + std::to_string(n_rows) +
            " x " + std::to_string(n_cols) + "], " +
            "but prob_mat is [" +
            std::to_string(prob_mat.size()) + " x " +
            std::to_string(prob_mat[0].size()) + " x " +
            std::to_string(prob_mat[0][0].size()) + "]. " +
            "Check if the number of provided parameters (e.g., 'd') exactly "
            "matches the number of condition levels."
        );
    }

/* ========================================================================== *
 *                    2. Numeric Stability Parameters                         *
 * ========================================================================== */
    // calc_tol 用于避免出现 log(0)，默认使用较小正值。
    double calc_tol = 1e-10;
    if (std_params.count("calc_tol") > 0 &&
        !std_params.at("calc_tol").empty()) {
        calc_tol = static_cast<double>(std_params.at("calc_tol")[0]);
    }

    std::vector<std::vector<std::vector<T>>> result(
        n_diffs, std::vector<std::vector<T>>(
            n_rows, std::vector<T>(n_cols, 0.0)
        )
    );

/* ========================================================================== *
 *                 3. Core Calculation: freq * log(prob)                      *
 * ========================================================================== */
    for (size_t d = 0; d < n_diffs; ++d) {
        for (size_t i = 0; i < n_rows; ++i) {
            for (size_t j = 0; j < n_cols; ++j) {
                T p = prob_mat[d][i][j];
                double freq = freq_mat[d][i][j];

                // 将概率压到安全区间，避免极端值造成数值错误。
                T p_adj =
                    (1.0 - calc_tol * static_cast<double>(n_cols)) * p +
                    calc_tol;

                using std::log;
                result[d][i][j] = freq * log(p_adj);
            }
        }
    }

    return result;
}

template std::vector<std::vector<std::vector<double>>> matrix_mult<double>(
    const std::vector<std::vector<std::vector<double>>>&,
    const std::vector<std::vector<std::vector<double>>>&,
    const std::unordered_map<std::string, std::vector<double>>&
);
