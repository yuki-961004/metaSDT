#ifndef CRITERION_PRIOR_HPP
#define CRITERION_PRIOR_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <numeric>

// ==============================================================================
// 核心类：模块化先验分布注入器 (Prior Injector)
// ==============================================================================
// 该类负责管理所有自由参数的先验概率分布 (Prior Distributions)。
// 它允许我们在贝叶斯推断或 MAP (最大后验) 估计中，为不同的参数独立挂载不同的概率密度函数。
class CriterionPrior {
public:
    // 定义支持的先验分布类型枚举
    enum class PriorType { 
        NORMAL, UNIFORM, LOGNORMAL, CAUCHY, BETA, EXPONENTIAL, NONE 
    };
    
    struct PriorSpec {
        PriorType type;
        double param1; // e.g., mu
        double param2; // e.g., sigma
    };
    
    // ==========================================================
    // 1. 先验映射表 (Prior Mapping Table)
    // ==========================================================
    // 存储从 Eigen 扁平参数向量的数值索引 (Index) 到对应分布参数 (PriorSpec) 的映射关系。
    std::unordered_map<int, PriorSpec> prior_specs_;

    void add_prior(int param_index, PriorType type, double p1, double p2) {
        prior_specs_[param_index] = {type, p1, p2};
    }

    // ==========================================================
    // 2. 核心求值：评估总对数先验概率 (Evaluate Log-Prior)
    // ==========================================================
    // 遍历当前所有的自由参数探索值 free_params，根据其指定的分布计算对数概率，并进行累加求和。
    template <typename T>
    T evaluate(const Eigen::Matrix<T, Eigen::Dynamic, 1>& free_params) const {
        T log_prior = 0.0;
        
        using std::log;
        using std::pow;
        const double PI = 3.14159265358979323846;

        for (const auto& kv : prior_specs_) {
            int idx = kv.first;
            const auto& spec = kv.second;
            
            // 防御性越界检查：如果索引超出了当前正在探索的参数维度，则安全跳过
            if (idx >= free_params.size()) continue;
            T val = free_params(idx);
            
            switch (spec.type) {
                case PriorType::NORMAL: {
                    // 计算正态分布的对数概率密度 (Log PDF of Normal Distribution)
                    // 注：在单纯的优化中忽略常数项 -0.5 * log(2*pi*sigma^2) 是可行的，
                    // 但为了未来与 MCMC 采样器 (如 Stan/NUTS) 完美对齐，保留严格的数学解析式。
                    double mu = spec.param1;
                    double sigma = spec.param2;
                    log_prior += -0.5 * log(2.0 * PI * sigma * sigma) 
                                 - 0.5 * pow((val - mu) / sigma, 2);
                    break;
                }
                case PriorType::LOGNORMAL: {
                    double mu = spec.param1;
                    double sigma = spec.param2;
                    if (val > 0.0) {
                        log_prior += -log(val * sigma * sqrt(2.0 * PI)) 
                                     - 0.5 * pow((log(val) - mu) / sigma, 2);
                    } else {
                        log_prior += -1e10; // 严谨的超出支持域边界惩罚
                    }
                    break;
                }
                case PriorType::CAUCHY: {
                    double location = spec.param1;
                    double scale = spec.param2;
                    log_prior += -log(PI * scale) 
                                 - log(1.0 + pow((val - location) / scale, 2));
                    break;
                }
                case PriorType::UNIFORM: {
                    double min_val = spec.param1;
                    double max_val = spec.param2;
                    if (val >= min_val && val <= max_val) {
                        log_prior += -log(max_val - min_val);
                    } else {
                        log_prior += -1e10;
                    }
                    break;
                }
                case PriorType::BETA: {
                    double shape1 = spec.param1;
                    double shape2 = spec.param2;
                    if (val > 0.0 && val < 1.0) {
                        // 严格加上 Beta 分布的归一化常数：lgamma(a) + lgamma(b) - lgamma(a+b)
                        double log_beta = std::lgamma(shape1) + 
                                          std::lgamma(shape2) - 
                                          std::lgamma(shape1 + shape2);
                        log_prior += (shape1 - 1.0) * log(val) + 
                                     (shape2 - 1.0) * log(1.0 - val) - log_beta;
                    } else {
                        log_prior += -1e10;
                    }
                    break;
                }
                case PriorType::EXPONENTIAL: {
                    double rate = spec.param1;
                    if (val >= 0.0) {
                        log_prior += log(rate) - rate * val;
                    } else {
                        log_prior += -1e10;
                    }
                    break;
                }
                default:
                    break;
            }
        }
        return log_prior;
    }

    // ==========================================================
    // 3. 经验贝叶斯核心：基于群体推断更新先验 (EM-MAP M-Step)
    // ==========================================================
    // 接收所有被试在当前迭代下估计出的自由参数矩阵 (N_subjects x K_params)
    // 使用矩估计 (Method of Moments) 或鲁棒统计在原生参数空间下更新超参数。
    void update(const std::vector<std::vector<double>>& group_free_params) {
        if (group_free_params.empty()) return;
        size_t n_subjects = group_free_params.size();
        
        // 如果少于2个被试，无法估计群体方差，直接跳过更新
        if (n_subjects < 2) return; 
        
        size_t k_params = group_free_params[0].size();

        for (auto& kv : prior_specs_) {
            int j = kv.first;
            if (j >= static_cast<int>(k_params)) continue;
            
            PriorSpec& spec = kv.second;
            // 均匀分布通常用于代表无信息边界，不参与群体推断的更新
            if (spec.type == PriorType::NONE || spec.type == PriorType::UNIFORM) {
                continue;
            }

            // 提取所有被试的第 j 个参数
            std::vector<double> x(n_subjects);
            for (size_t i = 0; i < n_subjects; ++i) {
                x[i] = group_free_params[i][j];
            }

            // 计算均值和无偏样本方差
            double mean = std::accumulate(x.begin(), x.end(), 0.0) / n_subjects;
            double var = 0.0;
            for (double v : x) var += (v - mean) * (v - mean);
            var /= (n_subjects - 1.0);

            // 设定最小容差，防止方差坍缩导致除以0或密度函数爆炸
            double min_var = 1e-6; 
            if (var < min_var) var = min_var;

            // 核心分发：依据分布类型进行原生空间映射更新
            switch (spec.type) {
                case PriorType::NORMAL: {
                    spec.param1 = mean;
                    spec.param2 = std::sqrt(var);
                    break;
                }
                case PriorType::LOGNORMAL: {
                    std::vector<double> log_x(n_subjects);
                    double log_sum = 0.0;
                    for (size_t i = 0; i < n_subjects; ++i) {
                        double val = x[i] > 1e-8 ? x[i] : 1e-8; // 防御性保护
                        log_x[i] = std::log(val);
                        log_sum += log_x[i];
                    }
                    double log_mean = log_sum / n_subjects;
                    double log_var = 0.0;
                    for (double v : log_x) log_var += (v - log_mean) * (v - log_mean);
                    log_var /= (n_subjects - 1.0);
                    
                    spec.param1 = log_mean;
                    spec.param2 = std::sqrt(std::max(log_var, min_var));
                    break;
                }
                case PriorType::BETA: {
                    // 防御：强制裁剪均值，防止处于绝对的 0 或 1 导致计算崩溃
                    double m = std::max(1e-4, std::min(mean, 1.0 - 1e-4));
                    // Beta 分布的方差必须严格小于 m * (1 - m)
                    double max_var = m * (1.0 - m) - 1e-6;
                    if (var > max_var) var = max_var; 

                    double common_term = (m * (1.0 - m) / var) - 1.0;
                    double alpha = m * common_term;
                    double beta = (1.0 - m) * common_term;

                    spec.param1 = std::max(alpha, 1e-3);
                    spec.param2 = std::max(beta, 1e-3);
                    break;
                }
                case PriorType::EXPONENTIAL: {
                    double rate = 1.0 / std::max(mean, 1e-6);
                    spec.param1 = rate;
                    break;
                }
                case PriorType::CAUCHY: {
                    // 柯西分布没有均值和方差，使用中位数 (Median) 和四分位距的一半 (IQR/2)
                    std::sort(x.begin(), x.end());
                    spec.param1 = (n_subjects % 2 == 0) ? 
                        (x[n_subjects / 2 - 1] + x[n_subjects / 2]) / 2.0 : 
                        x[n_subjects / 2];
                    double iqr = x[n_subjects * 3 / 4] - x[n_subjects / 4];
                    spec.param2 = std::max(iqr / 2.0, 1e-4);
                    break;
                }
                default: break;
            }
        }
    }
};

#endif // CRITERION_PRIOR_HPP