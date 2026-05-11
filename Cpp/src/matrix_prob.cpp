#include "../include/matrix_prob.hpp"
#include <stdexcept>

template <typename T>
MatrixProb<T> matrix_prob(
    const std::vector<T>& cdf_noise,
    const std::vector<T>& cdf_signal,
    const std::unordered_map<std::string, std::vector<T>>& std_params
) {
    // ==========================================================
    // 1. 安全检查与基础变量设定
    // ==========================================================
    if (cdf_noise.empty() || cdf_signal.empty()) {
        throw std::invalid_argument("Error: CDF vectors must not be empty.");
    }
    if (cdf_noise.size() != cdf_signal.size()) {
        throw std::invalid_argument(
            "Error: cdf_noise and cdf_signal must have the same length."
        );
    }

    size_t n_criteria = cdf_noise.size();
    size_t n_cols = n_criteria + 1; // 判定标准将分布切割为 N+1 个反应区间

    // ==========================================================
    // 2. 参数对齐验证与拦截
    // ==========================================================
    // 拦截：防止用户仅传入单点 CDF，却妄图生成带有复杂置信度的完整概率矩阵
    auto it_c_conf = std_params.find("c_conf");
    if (it_c_conf != std_params.end() && !it_c_conf->second.empty()) {
        size_t n_c_conf = it_c_conf->second.size();
        size_t expected_criteria;
        
        // 使用 find 替代多次 count + at
        auto it_n_conf = std_params.find("n_conf");
        if (it_n_conf != std_params.end() && !it_n_conf->second.empty() && 
            static_cast<size_t>(it_n_conf->second[0]) == n_c_conf) {
            // 此时 c_conf 已经是一个包含所有边界的完整向量
            expected_criteria = n_c_conf; 
        } else {
            // 1个反应边界 + 左右各 n_c_conf 个置信度边界
            expected_criteria = 1 + n_c_conf * 2; 
        }
        
        if (n_criteria != expected_criteria) {
            throw std::invalid_argument(
                "Error: Mismatch between criteria and c_conf. You provided " + 
                std::to_string(n_c_conf) + " c_conf values, " +
                "which requires exactly " + std::to_string(expected_criteria) + 
                " criteria points (CDFs) to split into " + 
                std::to_string(expected_criteria + 1) + " bins. " +
                "But CDF vectors only have " + std::to_string(n_criteria) + 
                " points. " +
                "Did you forget to pass the FULL criteria vector to model_sdt?"
            );
        }
    }

    MatrixProb<T> result;
    result.prob_mat.assign(2, std::vector<T>(n_cols, 0.0));

    // SDT 中的判断选项通常是偶数（例如 2键辨别, 或 2键x3置信度=6选项）
    if (n_cols % 2 != 0) {
        throw std::invalid_argument(
            "Error: Number of resulting intervals must be even (symmetric)."
        );
    }
    size_t n_conf = n_cols / 2;

    // ==========================================================
    // 3. 核心概率计算 (通过相邻 CDF 的面积差值计算区间概率)
    // ==========================================================
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

    // ==========================================================
    // 4. 全局变异修正 (引入按错键/走神概率 rate_lapse)
    // ==========================================================
    T lapse = 0.0;
    auto it_lapse = std_params.find("rate_lapse");
    if (it_lapse != std_params.end() && !it_lapse->second.empty()) {
        lapse = it_lapse->second[0];
    }

    if (lapse > 0.0) {
        for (int i = 0; i < 2; ++i) {
            for (size_t j = 0; j < n_cols; ++j) {
                // 反应概率 = 完全随机猜测的均等概率 + 认真作答的概率
                result.prob_mat[i][j] = 
                    (lapse / static_cast<T>(n_cols)) + 
                    ((1.0 - lapse) * result.prob_mat[i][j]);
            }
        }
    }

    // ==========================================================
    // 5. 元数据组装 (赋予与 matrix_freq 完美对齐的行列名)
    // ==========================================================
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

template MatrixProb<double> matrix_prob<double>(
    const std::vector<double>&, 
    const std::vector<double>&, 
    const std::unordered_map<std::string, std::vector<double>>&);