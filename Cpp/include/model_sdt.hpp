#ifndef MODEL_SDT_HPP
#define MODEL_SDT_HPP

#include <string>
#include <vector>
#include <unordered_map>

// 标准信号检测论模型 (Standard SDT)
template <typename T>
class ModelSDT {
private:
    // 内部核心参数
    T d;
    T sd_noise;
    T sd_signal;
    T mu_noise;
    T mu_signal;
    std::vector<T> criteria; // 自动生成的切割点

public:
    // 构造函数：只接收一个被拍扁的 params 字典
    ModelSDT(const std::unordered_map<std::string, std::vector<T>>& std_params);

    // 获取生成的切割点
    const std::vector<T>& get_criteria() const { return criteria; }

    // ==========================================================
    // 输出的两个累积分布函数 (CDF)
    // ==========================================================
    // 默认无参版本，使用内部自动生成的 criteria
    std::vector<T> cdf_noise() const;
    std::vector<T> cdf_signal() const;

    // 支持传入单个 x 值计算
    T cdf_noise(T x) const;
    T cdf_signal(T x) const;

    // 支持传入 x 向量批量计算
    std::vector<T> cdf_noise(const std::vector<T>& x_vec) const;
    std::vector<T> cdf_signal(const std::vector<T>& x_vec) const;
};

#endif // MODEL_SDT_HPP