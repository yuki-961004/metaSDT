#include "../include/matrix_prob.hpp"
#include <stdexcept>

/* ========================================================================== *
 *                            概率矩阵构建模块                                  *
 * ========================================================================== */

template <typename T>
MatrixProb<T> matrix_prob(
    const std::vector<std::vector<T>>& cdf_noise,
    const std::vector<std::vector<T>>& cdf_signal,
    const std::unordered_map<std::string, std::vector<T>>& std_params
) {
    /* ====================================================================== *
     * 1. 输入校验与基础维度定义
     * ====================================================================== */
    if (cdf_noise.empty() || cdf_signal.empty()) {
        throw std::invalid_argument("Error: CDF vectors must not be empty.");
    }
    if (cdf_noise.size() != cdf_signal.size()) {
        throw std::invalid_argument(
            "Error: cdf_noise and cdf_signal must have the same length."
        );
    }

    size_t n_diffs = cdf_noise.size();
    if (cdf_noise[0].empty() || cdf_signal[0].empty()) {
        throw std::invalid_argument("Error: CDF vectors must not be empty.");
    }
    size_t n_criteria = cdf_noise[0].size();
    size_t n_cols = n_criteria + 1;

    /* ====================================================================== *
     * 2. 参数一致性检查
     * ====================================================================== */
    auto it_c_conf = std_params.find("c_conf");
    if (it_c_conf != std_params.end() && !it_c_conf->second.empty()) {
        size_t n_c_conf = it_c_conf->second.size();
        size_t expected_criteria;

        auto it_n_conf = std_params.find("n_conf");
        if (it_n_conf != std_params.end() && !it_n_conf->second.empty() &&
            static_cast<size_t>(it_n_conf->second[0]) == n_c_conf) {
            expected_criteria = n_c_conf;
        } else {
            expected_criteria = 1 + n_c_conf * 2;
        }

        if (n_criteria != expected_criteria) {
            throw std::invalid_argument(
                "Error: Mismatch between criteria and c_conf. You provided " +
                std::to_string(n_c_conf) + " c_conf values, " +
                "which requires exactly " +
                std::to_string(expected_criteria) +
                " criteria points (CDFs) to split into " +
                std::to_string(expected_criteria + 1) + " bins. " +
                "But CDF vectors only have " + std::to_string(n_criteria) +
                " points. " +
                "Did you forget to pass the FULL criteria vector to "
                "model_sdt?"
            );
        }
    }

    MatrixProb<T> result;
    result.prob_mat.assign(
        n_diffs,
        std::vector<std::vector<T>>(2, std::vector<T>(n_cols, 0.0))
    );

    if (n_cols % 2 != 0) {
        throw std::invalid_argument(
            "Error: Number of resulting intervals must be even (symmetric)."
        );
    }
    size_t n_conf = n_cols / 2;

    /* ====================================================================== *
     * 3. 通过 CDF 差分构建区间概率
     * ====================================================================== */
    for (size_t d = 0; d < n_diffs; ++d) {
        result.prob_mat[d][0][0] = cdf_noise[d][0];
        result.prob_mat[d][1][0] = cdf_signal[d][0];

        for (size_t i = 1; i < n_criteria; ++i) {
            result.prob_mat[d][0][i] = cdf_noise[d][i] - cdf_noise[d][i - 1];
            result.prob_mat[d][1][i] = cdf_signal[d][i] - cdf_signal[d][i - 1];
        }

        result.prob_mat[d][0][n_cols - 1] =
            1.0 - cdf_noise[d][n_criteria - 1];
        result.prob_mat[d][1][n_cols - 1] =
            1.0 - cdf_signal[d][n_criteria - 1];
    }

    /* ====================================================================== *
     * 4. lapse 修正
     * ====================================================================== */
    T lapse = 0.0;
    auto it_lapse = std_params.find("rate_lapse");
    if (it_lapse != std_params.end() && !it_lapse->second.empty()) {
        lapse = it_lapse->second[0];
    }

    if (lapse > 0.0) {
        for (size_t d = 0; d < n_diffs; ++d) {
            for (int i = 0; i < 2; ++i) {
                for (size_t j = 0; j < n_cols; ++j) {
                    result.prob_mat[d][i][j] =
                        (lapse / static_cast<T>(n_cols)) +
                        ((1.0 - lapse) * result.prob_mat[d][i][j]);
                }
            }
        }
    }

    /* ====================================================================== *
     * 5. 元数据命名
     * ====================================================================== */
    result.row_names = {"stim_0", "stim_1"};

    if (n_cols == 2) {
        result.col_names = {"resp_0", "resp_1"};
    } else {
        for (size_t c = n_conf; c > 0; --c) {
            result.col_names.push_back("resp_0_conf_" + std::to_string(c));
        }
        for (size_t c = 1; c <= n_conf; ++c) {
            result.col_names.push_back("resp_1_conf_" + std::to_string(c));
        }
    }

    return result;
}

template MatrixProb<double> matrix_prob<double>(
    const std::vector<std::vector<double>> &,
    const std::vector<std::vector<double>> &,
    const std::unordered_map<std::string, std::vector<double>>&
);
