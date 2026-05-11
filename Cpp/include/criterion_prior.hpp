#ifndef CRITERION_PRIOR_HPP
#define CRITERION_PRIOR_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <cmath>

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
                        double log_beta = std::lgamma(shape1) + std::lgamma(shape2) 
                                          - std::lgamma(shape1 + shape2);
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
};

#endif // CRITERION_PRIOR_HPP