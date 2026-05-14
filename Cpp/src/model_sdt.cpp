#include "../include/model_sdt.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================== *
 *                         ModelSDT Construction                              *
 * ========================================================================== */

// 构造 SDT 模型对象，并从标准参数映射中提取核心参数。
// 该过程同时完成：
// 1) 均值向量初始化。
// 2) 准则点（criteria）构建。
// 3) 可选参数（如 sort_d、c_resp、c_conf）处理。
template <typename T>
ModelSDT<T>::ModelSDT(
    const std::unordered_map<std::string, std::vector<T>>& std_params
) {
    try {
        // d 为每个难度维度对应的敏感度参数向量。
        d_vec = std_params.at("d");

        // 噪声分布标准差取标量首元素。
        sd_noise = std_params.at("sd_noise")[0];

        // 信号分布标准差取标量首元素。
        sd_signal = std_params.at("sd_signal")[0];
    } catch (const std::out_of_range& e) {
        // 若缺少核心参数，直接抛出明确初始化错误。
        throw std::invalid_argument(
            "ModelSDT Initialization Error: Missing required parameters "
            "('d', 'sd_noise', or 'sd_signal')."
        );
    }

    // 可选参数 sort_d 控制是否对 d 做降序排序。
    T sort_d = 0.0;
    if (std_params.count("sort_d") && !std_params.at("sort_d").empty()) {
        sort_d = std_params.at("sort_d")[0];
    }

    // 当 sort_d 非 0 时，按降序排列 d，保证维度顺序一致。
    if (sort_d != 0.0) {
        std::sort(d_vec.rbegin(), d_vec.rend());
    }

    // 对每个 d 维度构造噪声与信号分布的均值。
    // SDT 常见参数化：mu_noise=-d/2，mu_signal=+d/2。
    for (size_t i = 0; i < d_vec.size(); ++i) {
        mu_noise_vec.push_back(-d_vec[i] / 2.0);
        mu_signal_vec.push_back(d_vec[i] / 2.0);
    }

    // 默认响应准则中心点为 0；若有 c_resp 则使用用户值。
    T c_resp_val = 0.0;
    auto it_c_resp = std_params.find("c_resp");
    if (it_c_resp != std_params.end() && !it_c_resp->second.empty()) {
        c_resp_val = it_c_resp->second[0];
    }

    // c_conf 用于构造置信度相关的多阈值准则点。
    auto it_c_conf = std_params.find("c_conf");
    if (it_c_conf != std_params.end() && !it_c_conf->second.empty()) {
        // 复制并排序，确保阈值从小到大有序。
        std::vector<T> c_conf = it_c_conf->second;
        std::sort(c_conf.begin(), c_conf.end());

        // n_conf 可用于声明 c_conf 是否已是完整阈值向量。
        auto it_n_conf = std_params.find("n_conf");
        bool has_n_conf = (it_n_conf != std_params.end() &&
                           !it_n_conf->second.empty());

        // 当 n_conf 与 c_conf 长度一致时，直接把 c_conf 作为准则点。
        bool is_full_vector = (
            has_n_conf &&
            static_cast<int>(it_n_conf->second[0]) ==
            static_cast<int>(c_conf.size())
        );

        if (is_full_vector) {
            // c_conf 已完整给出，直接赋值。
            criteria = c_conf;
        } else {
            // 否则围绕 c_resp 对称扩展。
            // 左侧为 c_resp-c_conf(逆序)。
            // 中间为 c_resp。
            // 右侧为 c_resp+c_conf。
            criteria.reserve(1 + c_conf.size() * 2);
            for (auto it = c_conf.rbegin(); it != c_conf.rend(); ++it) {
                criteria.push_back(c_resp_val - *it);
            }
            criteria.push_back(c_resp_val);
            for (auto it = c_conf.begin(); it != c_conf.end(); ++it) {
                criteria.push_back(c_resp_val + *it);
            }
        }
    } else {
        // 若无 c_conf，仅使用单一响应准则点 c_resp。
        criteria.push_back(c_resp_val);
    }
}

/* ========================================================================== *
 *                           CDF Matrix Builders                              *
 * ========================================================================== */

// 逐维度计算噪声分布在全部准则点上的 CDF 矩阵。
template <typename T>
std::vector<std::vector<T>> ModelSDT<T>::cdf_noise() const {
    std::vector<std::vector<T>> res(d_vec.size());

    // 每个维度单独调用向量版 cdf_noise，保证维度可独立处理。
    for (size_t i = 0; i < d_vec.size(); ++i) {
        res[i] = cdf_noise(/*x_vec=*/this->criteria, /*dim_idx=*/i);
    }
    return res;
}

// 逐维度计算信号分布在全部准则点上的 CDF 矩阵。
template <typename T>
std::vector<std::vector<T>> ModelSDT<T>::cdf_signal() const {
    std::vector<std::vector<T>> res(d_vec.size());

    // 每个维度单独调用向量版 cdf_signal，接口与噪声侧保持对称。
    for (size_t i = 0; i < d_vec.size(); ++i) {
        res[i] = cdf_signal(/*x_vec=*/this->criteria, /*dim_idx=*/i);
    }
    return res;
}

/* ========================================================================== *
 *                          Scalar CDF Evaluation                             *
 * ========================================================================== */

// 计算噪声分布在给定 x（指定维度）处的正态 CDF。
template <typename T>
T ModelSDT<T>::cdf_noise(T x, size_t dim_idx) const {
    using std::erf;
    using std::sqrt;

    // 正态分布 CDF 与误差函数关系：
    // Phi(z)=0.5*(1+erf(z/sqrt(2)))。
    return 0.5 * (1.0 + erf(
        (x - mu_noise_vec[dim_idx]) / (sd_noise * sqrt(2.0))
    ));
}

// 计算信号分布在给定 x（指定维度）处的正态 CDF。
template <typename T>
T ModelSDT<T>::cdf_signal(T x, size_t dim_idx) const {
    using std::erf;
    using std::sqrt;

    // 与噪声侧公式相同，仅均值和标准差来自信号分布参数。
    return 0.5 * (1.0 + erf(
        (x - mu_signal_vec[dim_idx]) / (sd_signal * sqrt(2.0))
    ));
}

/* ========================================================================== *
 *                          Vector CDF Evaluation                             *
 * ========================================================================== */

// 计算噪声分布在一组 x 点（指定维度）上的 CDF 向量。
template <typename T>
std::vector<T> ModelSDT<T>::cdf_noise(
    const std::vector<T>& x_vec,
    size_t dim_idx
) const {
    std::vector<T> y_vec(x_vec.size());

    // 逐点调用标量版本，复用统一公式并减少重复实现风险。
    for (size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_noise(/*x=*/x_vec[i], /*dim_idx=*/dim_idx);
    }
    return y_vec;
}

// 计算信号分布在一组 x 点（指定维度）上的 CDF 向量。
template <typename T>
std::vector<T> ModelSDT<T>::cdf_signal(
    const std::vector<T>& x_vec,
    size_t dim_idx
) const {
    std::vector<T> y_vec(x_vec.size());

    // 逐点调用标量版本，保证信号侧与噪声侧行为一致。
    for (size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_signal(/*x=*/x_vec[i], /*dim_idx=*/dim_idx);
    }
    return y_vec;
}

// 显式实例化 double 版本，供链接阶段直接使用。
template class ModelSDT<double>;
