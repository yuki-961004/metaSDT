#include "../include/model_sdt.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

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
template <typename T>
ModelSDT<T>::ModelSDT(
    const std::unordered_map<std::string, std::vector<T>>& std_params
) {
    
    // ==========================================================
    // 1.1 基础参数提取与校验
    // ==========================================================
    try {
        d_vec = std_params.at("d"); // 提取整个 d 的向量
        sd_noise = std_params.at("sd_noise")[0];
        sd_signal = std_params.at("sd_signal")[0];
    } catch (const std::out_of_range& e) {
        throw std::invalid_argument(
          "ModelSDT Initialization Error: Missing required parameters "
          "('d', 'sd_noise', or 'sd_signal')."
        );
    }

    // 探测是否开启对 d 的排序
    T sort_d = 0.0;
    if (std_params.count("sort_d") && !std_params.at("sort_d").empty()) {
        sort_d = std_params.at("sort_d")[0];
    }
    if (sort_d != 0.0) {
        // difficulty 越小 (例如 1) 代表越容易，d 应该越大。
        // 所以这里直接降序排列，使得 d_vec[0] (最大) 对齐 difficulty 1
        std::sort(d_vec.rbegin(), d_vec.rend());
    }

    // ==========================================================
    // 1.2 信号与噪声分布的均值计算
    // ==========================================================
    for (size_t i = 0; i < d_vec.size(); ++i) {
        mu_noise_vec.push_back(-d_vec[i] / 2.0);
        mu_signal_vec.push_back(d_vec[i] / 2.0);
    }

    // ==========================================================
    // 1.3 判定标准 (Criteria) 的动态生成机制
    // ==========================================================
    T c_resp_val = 0.0;
    auto it_c_resp = std_params.find("c_resp");
    if (it_c_resp != std_params.end() && !it_c_resp->second.empty()) {
        c_resp_val = it_c_resp->second[0];
    }

    // 冗余优化：利用 find 迭代器避免多次 count 和 at 的哈希表查询开销
    auto it_c_conf = std_params.find("c_conf");
    if (it_c_conf != std_params.end() && !it_c_conf->second.empty()) {
        std::vector<T> c_conf = it_c_conf->second;
        
        // 【核心防御】：无论优化器以何种顺序随机探索边界，
        // 在计算概率前强制排序，严格保证所有切割点的数学单调性！
        // 这彻底消除了对 NLOPT 复杂不等式约束的依赖。
        std::sort(c_conf.begin(), c_conf.end());
        
        auto it_n_conf = std_params.find("n_conf");
        bool has_n_conf = (it_n_conf != std_params.end() && 
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
// 2. 累积分布函数 (无参批量版，返回 2D 数组: [dim][criteria])
// ==========================================================
template <typename T>
std::vector<std::vector<T>> ModelSDT<T>::cdf_noise() const {
    std::vector<std::vector<T>> res(d_vec.size());
    for (size_t i = 0; i < d_vec.size(); ++i) {
        res[i] = cdf_noise(/*x_vec=*/this->criteria, /*dim_idx=*/i);
    }
    return res;
}

template <typename T>
std::vector<std::vector<T>> ModelSDT<T>::cdf_signal() const {
    std::vector<std::vector<T>> res(d_vec.size());
    for (size_t i = 0; i < d_vec.size(); ++i) {
        res[i] = cdf_signal(/*x_vec=*/this->criteria, /*dim_idx=*/i);
    }
    return res;
}

// ==========================================================
// 3. 累积分布函数 (标量版：针对单个数据点和特定维度)
// ==========================================================
template <typename T>
T ModelSDT<T>::cdf_noise(T x, size_t dim_idx) const {
    // ADL (Argument-Dependent Lookup) 卫生规范
    using std::erf;
    using std::sqrt;
    return 0.5 * (1.0 + erf(
        (x - mu_noise_vec[dim_idx]) / (sd_noise * sqrt(2.0))
    ));
}

template <typename T>
T ModelSDT<T>::cdf_signal(T x, size_t dim_idx) const {
    using std::erf;
    using std::sqrt;
    return 0.5 * (1.0 + erf(
        (x - mu_signal_vec[dim_idx]) / (sd_signal * sqrt(2.0))
    ));
}

// ==========================================================
// 4. 累积分布函数 (向量版)
// ==========================================================
template <typename T>
std::vector<T> ModelSDT<T>::cdf_noise(
    const std::vector<T>& x_vec, size_t dim_idx
) const {
    std::vector<T> y_vec(x_vec.size());
    for (size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_noise(/*x=*/x_vec[i], /*dim_idx=*/dim_idx);
    }
    return y_vec;
}

template <typename T>
std::vector<T> ModelSDT<T>::cdf_signal(
    const std::vector<T>& x_vec, size_t dim_idx
) const {
    std::vector<T> y_vec(x_vec.size());
    for (size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_signal(/*x=*/x_vec[i], /*dim_idx=*/dim_idx);
    }
    return y_vec;
}

// 显式实例化 double 类型，保证分离编译时不报错，完美对接 NLOPT
template class ModelSDT<double>;