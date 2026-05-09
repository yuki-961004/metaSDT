#include "../include/loss_function.hpp"
#include <cmath>
#include <stdexcept>

LossResult loss_function(
    const std::vector<std::vector<double>>& mult_mat,
    const std::vector<std::vector<double>>& freq_mat,
    int k
) {
    // ==========================================================
    // 1. 安全检查
    // ==========================================================
    if (mult_mat.empty() || freq_mat.empty()) {
        throw std::invalid_argument("Error: Input matrices cannot be empty.");
    }

    // ==========================================================
    // 2. 遍历矩阵：计算总对数似然 (LogL) 与总试次 (N)
    // ==========================================================
    double logL = 0.0;
    double N = 0.0;

    for (size_t i = 0; i < mult_mat.size(); ++i) {
        for (size_t j = 0; j < mult_mat[i].size(); ++j) {
            logL += mult_mat[i][j];
            N += freq_mat[i][j];
        }
    }

    // ==========================================================
    // 3. 封装计算信息准则指标 (AIC, BIC 等)
    // ==========================================================
    LossResult res;
    res.logL = logL;
    res.nll = -logL; // 供 NLOPT 使用的最小化目标
    res.k = k;
    res.aic = 2.0 * k - 2.0 * logL;
    
    if (N > 0.0) {
        res.bic = k * std::log(N) - 2.0 * logL;
    } else {
        res.bic = 0.0; 
    }
    return res;
}