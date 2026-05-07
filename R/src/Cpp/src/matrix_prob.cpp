#include "../include/matrix_prob.hpp"
#include <stdexcept>

MatrixProb matrix_prob(
    const std::vector<double>& cdf_noise,
    const std::vector<double>& cdf_signal,
    const std::unordered_map<std::string, std::vector<double>>& params
) {
    // 安全检查：确保传入了 CDF 结果
    if (cdf_noise.empty() || cdf_signal.empty()) {
        throw std::invalid_argument("Error: CDF vectors must not be empty.");
    }
    if (cdf_noise.size() != cdf_signal.size()) {
        throw std::invalid_argument("Error: cdf_noise and cdf_signal must have the same length.");
    }

    size_t n_criteria = cdf_noise.size();
    size_t n_cols = n_criteria + 1; // N个判定标准会将分布切割为 N+1 个反应区间

    // =========================================================
    // 拦截检查：防止用户只传入了单点 CDF 却期望得到完整的概率矩阵
    // =========================================================
    if (params.count("c_conf") > 0 && !params.at("c_conf").empty()) {
        size_t n_c_conf = params.at("c_conf").size();
        size_t expected_criteria = 1 + n_c_conf * 2; // 1个反应边界 + 左右各 n_c_conf 个置信度边界
        
        if (n_criteria != expected_criteria) {
            throw std::invalid_argument(
                "Error: Mismatch between criteria and c_conf. You provided " + std::to_string(n_c_conf) + 
                " c_conf values, which requires exactly " + std::to_string(expected_criteria) + 
                " criteria points (CDFs) to split into " + std::to_string(expected_criteria + 1) + " bins. " +
                "But CDF vectors only have " + std::to_string(n_criteria) + " points. " +
                "Did you forget to pass the FULL criteria vector to model_sdt?"
            );
        }
    }

    MatrixProb result;
    result.prob_mat.assign(2, std::vector<double>(n_cols, 0.0));

    // 为了处理置信度，我们假定响应是对称的 (即一半是噪音，一半是信号)
    if (n_cols % 2 != 0) {
        throw std::invalid_argument("Error: Number of resulting intervals must be even (symmetric responses).");
    }
    size_t n_conf = n_cols / 2;

    // =========================================================
    // 1. 核心概率计算 (支持任意数量的判定标准，通过计算面积差值)
    // =========================================================
    // 按照 X 轴从左到右 (321123 的自然顺序) 直接填充概率矩阵

    // 第一个区间: P(X < c_1)
    result.prob_mat[0][0] = cdf_noise[0];
    result.prob_mat[1][0] = cdf_signal[0];

    // 中间的各个区间: P(c_i < X < c_{i+1}) = CDF(c_{i+1}) - CDF(c_i)
    for (size_t i = 1; i < n_criteria; ++i) {
        result.prob_mat[0][i] = cdf_noise[i] - cdf_noise[i - 1];
        result.prob_mat[1][i] = cdf_signal[i] - cdf_signal[i - 1];
    }

    // 最后一个区间: P(X > c_N) = 1.0 - CDF(c_N)
    result.prob_mat[0][n_cols - 1] = 1.0 - cdf_noise[n_criteria - 1];
    result.prob_mat[1][n_cols - 1] = 1.0 - cdf_signal[n_criteria - 1];

    // =========================================================
    // 2. 结合 params 中的全局参数 (引入 rate_lapse 修正)
    // =========================================================
    double lapse = 0.0;
    if (params.count("rate_lapse") > 0 && !params.at("rate_lapse").empty()) {
        lapse = params.at("rate_lapse")[0];
    }

    if (lapse > 0.0) {
        for (int i = 0; i < 2; ++i) {
            for (size_t j = 0; j < n_cols; ++j) {
                // 真实的反应概率 = 完全随机猜测的概率(落入任何区间的概率均等) + 认真作答的概率
                result.prob_mat[i][j] = (lapse / static_cast<double>(n_cols)) + ((1.0 - lapse) * result.prob_mat[i][j]);
            }
        }
    }

    // =========================================================
    // 3. 赋予与 matrix_freq 完美对齐的列名
    // =========================================================
    result.row_names = {"stim_0", "stim_1"};
    
    if (n_cols == 2) {
        // 无置信度的标准 SDT
        result.col_names = {"resp_0", "resp_1"};
    } else {
        // 有置信度的情况，生成 321123 模式的列名
        for (size_t c = n_conf; c > 0; --c) {
            result.col_names.push_back("resp_0_conf_" + std::to_string(c));
        }
        for (size_t c = 1; c <= n_conf; ++c) {
            result.col_names.push_back("resp_1_conf_" + std::to_string(c));
        }
    }

    return result;
}