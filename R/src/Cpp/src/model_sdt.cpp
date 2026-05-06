#include "../include/model_sdt.hpp"
#include <cmath>
#include <stdexcept>

// 定义圆周率 (防止某些 Windows 编译器未默认定义 M_PI)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==========================================================
// 1. 模型初始化 (只接受 params)
// ==========================================================
ModelSDT::ModelSDT(const std::unordered_map<std::string, std::vector<double>>& params) {
    // 从拍扁后的参数字典中安全提取所需参数
    try {
        d = params.at("d")[0];
        sd_noise = params.at("sd_noise")[0];
        sd_signal = params.at("sd_signal")[0];
    } catch (const std::out_of_range& e) {
        throw std::invalid_argument(
          "ModelSDT Initialization Error: Missing required parameters ('d', 'sd_noise', or 'sd_signal')."
        );
    }

    // 计算两个分布的均值
    // 为了配合默认参数中的 c_resp = 0.0 (无偏好)，我们将噪声和信号分布对称放置在 0 的两侧
    mu_noise = -d / 2.0;
    mu_signal = d / 2.0;
}

// ==========================================================
// 2. 累积分布函数 (标量版)
// ==========================================================
double ModelSDT::cdf_noise(double x) const {
    // 正态分布 CDF 公式: 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    return 0.5 * (1.0 + std::erf((x - mu_noise) / (sd_noise * std::sqrt(2.0))));
}

double ModelSDT::cdf_signal(double x) const {
    return 0.5 * (1.0 + std::erf((x - mu_signal) / (sd_signal * std::sqrt(2.0))));
}

// ==========================================================
// 3. 累积分布函数 (向量批量版)
// ==========================================================
std::vector<double> ModelSDT::cdf_noise(const std::vector<double>& x_vec) const {
    std::vector<double> y_vec(x_vec.size());
    for (size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_noise(x_vec[i]);
    }
    return y_vec;
}

std::vector<double> ModelSDT::cdf_signal(const std::vector<double>& x_vec) const {
    std::vector<double> y_vec(x_vec.size());
    for (size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_signal(x_vec[i]);
    }
    return y_vec;
}