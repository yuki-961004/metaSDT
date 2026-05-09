#include "../include/model_sdt.hpp"
#include <cmath>
#include <stdexcept>

// ==========================================================
// 0. 宏定义与预处理
// ==========================================================
// 定义圆周率 (防止某些 Windows 编译器未默认定义 M_PI)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==========================================================
// 1. 模型初始化与核心参数提取 (只接受 params)
// ==========================================================
ModelSDT::ModelSDT(
    const std::unordered_map<std::string, std::vector<double>>& params
) {
    
    // ==========================================================
    // 1.1 基础参数提取与校验
    // ==========================================================
    try {
        d = params.at("d")[0];
        sd_noise = params.at("sd_noise")[0];
        sd_signal = params.at("sd_signal")[0];
    } catch (const std::out_of_range& e) {
        throw std::invalid_argument(
          "ModelSDT Initialization Error: Missing required parameters "
          "('d', 'sd_noise', or 'sd_signal')."
        );
    }

    // ==========================================================
    // 1.2 信号与噪声分布的均值计算
    // ==========================================================
    // 默认模型以 0 为绝对中心。
    // 无偏好时 (c_resp = 0.0)，噪声和信号对称分布在两侧
    mu_noise = -d / 2.0;
    mu_signal = d / 2.0;

    // ==========================================================
    // 1.3 判定标准 (Criteria) 的动态生成机制
    // ==========================================================
    double c_resp_val = 0.0;
    auto it_c_resp = params.find("c_resp");
    if (it_c_resp != params.end() && !it_c_resp->second.empty()) {
        c_resp_val = it_c_resp->second[0];
    }

    // 冗余优化：利用 find 迭代器避免多次 count 和 at 的哈希表查询开销
    auto it_c_conf = params.find("c_conf");
    if (it_c_conf != params.end() && !it_c_conf->second.empty()) {
        const auto& c_conf = it_c_conf->second;
        
        auto it_n_conf = params.find("n_conf");
        bool has_n_conf = (it_n_conf != params.end() && 
                           !it_n_conf->second.empty());
        
        // 传递了 n_conf 且等于 c_conf 长度，说明这是非等距完整向量
        bool is_full_vector = (
            has_n_conf && 
            static_cast<int>(it_n_conf->second[0]) == 
            static_cast<int>(c_conf.size())
        );

        if (is_full_vector) {
            // 情况 A：c_conf 已经是包含所有判断标准的不等距完整向量
            criteria = c_conf;
        } else {
            // 情况 B：c_conf 仅提供了对称置信度宽度，需结合 c_resp 镜像展开
            // 预分配内存，避免多次 push_back 导致的 vector 动态扩容
            criteria.reserve(1 + c_conf.size() * 2);
            
            // 1. 左侧边界 (噪声响应) -> 倒序遍历，保证 criteria 呈升序排列
            for (auto it = c_conf.rbegin(); it != c_conf.rend(); ++it) {
                criteria.push_back(c_resp_val - *it);
            }
            // 2. 中间的一阶决策边界
            criteria.push_back(c_resp_val);
            // 3. 右侧边界 (信号响应) -> 正序遍历
            for (auto it = c_conf.begin(); it != c_conf.end(); ++it) {
                criteria.push_back(c_resp_val + *it);
            }
        }
    } else {
        // 情况 C：无置信度参数，退化为经典 SDT 的单判断标准
        criteria.push_back(c_resp_val);
    }
}

// ==========================================================
// 2. 累积分布函数 (无参批量版，依赖内部自动生成的 criteria)
// ==========================================================
std::vector<double> ModelSDT::cdf_noise() const {
    return cdf_noise(this->criteria);
}

std::vector<double> ModelSDT::cdf_signal() const {
    return cdf_signal(this->criteria);
}

// ==========================================================
// 3. 累积分布函数 (标量版：针对单个数据点)
// ==========================================================
double ModelSDT::cdf_noise(double x) const {
    // 正态分布 CDF 标准公式：0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    return 0.5 * (1.0 + std::erf(
        (x - mu_noise) / (sd_noise * std::sqrt(2.0))
    ));
}

double ModelSDT::cdf_signal(double x) const {
    return 0.5 * (1.0 + std::erf(
        (x - mu_signal) / (sd_signal * std::sqrt(2.0))
    ));
}

// ==========================================================
// 4. 累积分布函数 (向量版：供外部传入自定义坐标轴向量使用)
// ==========================================================
std::vector<double> ModelSDT::cdf_noise(
    const std::vector<double>& x_vec
) const {
    std::vector<double> y_vec(x_vec.size());
    for (size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_noise(x_vec[i]);
    }
    return y_vec;
}

std::vector<double> ModelSDT::cdf_signal(
    const std::vector<double>& x_vec
) const {
    std::vector<double> y_vec(x_vec.size());
    for (size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_signal(x_vec[i]);
    }
    return y_vec;
}