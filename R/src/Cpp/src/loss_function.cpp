#include "../include/loss_function.hpp"
#include <cmath>
#include <stdexcept>

LossResult loss_function(
    const std::vector<std::vector<double>>& mult_mat,
    const std::vector<std::vector<double>>& freq_mat,
    const std::unordered_map<std::string, std::vector<double>>& params,
    const std::vector<std::string>& free_params
) {
    if (mult_mat.empty() || freq_mat.empty()) {
        throw std::invalid_argument("Error: Input matrices cannot be empty.");
    }

    double logL = 0.0;
    double N = 0.0;

    // 1. 遍历矩阵，同时累加总对数似然 (logL) 和总试次数 (N)
    for (size_t i = 0; i < mult_mat.size(); ++i) {
        for (size_t j = 0; j < mult_mat[i].size(); ++j) {
            logL += mult_mat[i][j];
            N += freq_mat[i][j];
        }
    }

    // 2. 精准计算自由参数个数 k
    // 遍历 free_params 列表，提取字典中的数组长度（支持多维度的 c_conf）
    int k = 0;
    for (const auto& param_name : free_params) {
        if (params.count(param_name) > 0) {
            k += params.at(param_name).size();
        }
    }

    // 3. 封装并计算所有的指标
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