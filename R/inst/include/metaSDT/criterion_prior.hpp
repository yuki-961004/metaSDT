#ifndef CRITERION_PRIOR_HPP
#define CRITERION_PRIOR_HPP

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

/* ========================================================================== *
 *                         Criterion Prior Engine                             *
 * ========================================================================== */

class CriterionPrior {
public:
    // 定义支持的先验概率分布类型
    enum class PriorType {
        NORMAL,
        UNIFORM,
        LOGNORMAL,
        CAUCHY,
        BETA,
        EXPONENTIAL,
        NONE
    };

    // 先验参数规格体
    struct PriorSpec {
        PriorType type;
        double param1;
        double param2;
    };

    std::unordered_map<int, PriorSpec> prior_specs_;

    // 为指定的参数索引注册先验配置
    void add_prior(int param_index, PriorType type, double p1, double p2) {
        prior_specs_[param_index] = {type, p1, p2};
    }

/* ========================================================================== *
 *                        Log-Prior Evaluation                                *
 * ========================================================================== */

    // 专为 MCMC (马尔可夫链蒙特卡洛) 采样设计的对数先验求值器
    // 给定当前的一组自由参数, 计算并返回其总的对数先验概率
    template <typename T>
    T evaluate(const Eigen::Matrix<T, Eigen::Dynamic, 1>& free_params) const {
        T log_prior = 0.0;

        using std::log;
        using std::pow;
        const double PI = 3.14159265358979323846;

        // 遍历所有已注册的先验配置
        for (const auto& kv : prior_specs_) {
            const int idx = kv.first;
            const auto& spec = kv.second;

            // 如果当前索引超出了参数向量范围, 则安全跳过
            if (idx >= free_params.size()) {
                continue;
            }
            const T val = free_params(idx);

            switch (spec.type) {
                // 正态分布 (Normal): param1=mean, param2=sd
                case PriorType::NORMAL: {
                    const double mu = spec.param1;
                    const double sigma = spec.param2;
                    log_prior += -0.5 * log(2.0 * PI * sigma * sigma)
                                 - 0.5 * pow((val - mu) / sigma, 2);
                    break;
                }
                // 对数正态分布 (Lognormal): param1=meanlog, param2=sdlog
                case PriorType::LOGNORMAL: {
                    const double mu = spec.param1;
                    const double sigma = spec.param2;
                    if (val > 0.0) {
                        // 对数正态分布的对数密度公式
                        log_prior += -log(val * sigma * sqrt(2.0 * PI))
                                     - 0.5 * pow((log(val) - mu) / sigma, 2);
                    } else {
                        // 严格惩罚非正值域
                        log_prior += -1e10;
                    }
                    break;
                }
                // 柯西分布 (Cauchy): param1=location, param2=scale
                case PriorType::CAUCHY: {
                    const double location = spec.param1;
                    const double scale = spec.param2;
                    log_prior += -log(PI * scale)
                                 - log(1.0 + pow((val - location) / scale, 2));
                    break;
                }
                // 均匀分布 (Uniform): param1=min, param2=max
                case PriorType::UNIFORM: {
                    const double min_val = spec.param1;
                    const double max_val = spec.param2;
                    if (val >= min_val && val <= max_val) {
                        // 常数概率密度
                        log_prior += -log(max_val - min_val);
                    } else {
                        log_prior += -1e10;
                    }
                    break;
                }
                // 贝塔分布 (Beta): param1=shape1, param2=shape2
                case PriorType::BETA: {
                    const double shape1 = spec.param1;
                    const double shape2 = spec.param2;
                    if (val > 0.0 && val < 1.0) {
                        // 贝塔分布要求值域严格在 (0, 1) 之间
                        const double log_beta = std::lgamma(shape1)
                                              + std::lgamma(shape2)
                                              - std::lgamma(shape1 + shape2);
                        log_prior += (shape1 - 1.0) * log(val)
                                   + (shape2 - 1.0) * log(1.0 - val)
                                   - log_beta;
                    } else {
                        log_prior += -1e10;
                    }
                    break;
                }
                // 指数分布 (Exponential): param1=rate
                case PriorType::EXPONENTIAL: {
                    const double rate = spec.param1;
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

/* ========================================================================== *
 *                        Empirical Prior Update                              *
 * ========================================================================== */

    // 专为 EM-MAP (最大后验期望最大化) 算法设计的经验先验更新器
    // 利用一组被试在当前迭代的最佳拟合参数, 更新群体级别的先验超参数
    void update(const std::vector<std::vector<double>>& group_free_params) {
        if (group_free_params.empty()) {
            return;
        }
        const size_t n_subjects = group_free_params.size();
        if (n_subjects < 2) {
            // 如果被试数量不足以计算方差, 则跳过更新
            return;
        }

        const size_t k_params = group_free_params[0].size();

        // 遍历所有注册的先验, 逐一提取跨被试的参数切片
        for (auto& kv : prior_specs_) {
            const int j = kv.first;
            if (j >= static_cast<int>(k_params)) {
                continue;
            }

            PriorSpec& spec = kv.second;
            if (spec.type == PriorType::NONE ||
                spec.type == PriorType::UNIFORM) {
                // 无先验或均匀分布的边界不参与经验贝叶斯更新
                continue;
            }

            // 提取当前参数在所有被试中的值
            std::vector<double> x(n_subjects);
            for (size_t i = 0; i < n_subjects; ++i) {
                x[i] = group_free_params[i][j];
            }

            // 计算跨被试的均值与无偏估计方差
            const double mean =
                std::accumulate(x.begin(), x.end(), 0.0) / n_subjects;
            double var = 0.0;
            for (const double v : x) {
                var += (v - mean) * (v - mean);
            }
            var /= (n_subjects - 1.0);

            // 设置一个极小方差下限, 避免除以零导致数值崩溃
            const double min_var = 1e-6;
            if (var < min_var) {
                var = min_var;
            }

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
                        const double val = x[i] > 1e-8 ? x[i] : 1e-8;
                        log_x[i] = std::log(val);
                        log_sum += log_x[i];
                    }
                    const double log_mean = log_sum / n_subjects;
                    double log_var = 0.0;
                    for (const double v : log_x) {
                        log_var += (v - log_mean) * (v - log_mean);
                    }
                    log_var /= (n_subjects - 1.0);

                    spec.param1 = log_mean;
                    spec.param2 = std::sqrt(std::max(log_var, min_var));
                    break;
                }
                case PriorType::BETA: {
                    // 矩估计 (Method of Moments) 更新 Beta 分布参数
                    // 约束方差不能超过理论上限, 避免 common_term 变为负数
                    const double m = std::max(
                        1e-4,
                        std::min(mean, 1.0 - 1e-4)
                    );
                    const double max_var = m * (1.0 - m) - 1e-6;
                    if (var > max_var) {
                        var = max_var;
                    }

                    const double common_term = (m * (1.0 - m) / var) - 1.0;
                    const double alpha = m * common_term;
                    const double beta = (1.0 - m) * common_term;

                    spec.param1 = std::max(alpha, 1e-3);
                    spec.param2 = std::max(beta, 1e-3);
                    break;
                }
                case PriorType::EXPONENTIAL: {
                    spec.param1 = 1.0 / std::max(mean, 1e-6);
                    break;
                }
                case PriorType::CAUCHY: {
                    std::sort(x.begin(), x.end());
                    spec.param1 = (n_subjects % 2 == 0)
                        ? (x[n_subjects / 2 - 1] + x[n_subjects / 2]) / 2.0
                        : x[n_subjects / 2];
                    const double iqr =
                        x[n_subjects * 3 / 4] - x[n_subjects / 4];
                    spec.param2 = std::max(iqr / 2.0, 1e-4);
                    break;
                }
                default:
                    break;
            }
        }
    }
};

/* ========================================================================== *
 *                          State Update Helpers                              *
 * ========================================================================== */

// 专为 EM-MAP 算法设计的初始状态同步辅助函数
// 将当前迭代找到的最佳拟合结果回写至任务对象, 作为下一次优化的初始值
template <typename TaskType, typename ResultType>
inline void optimal(TaskType& task, const ResultType& res) {
    for (const auto& name : task.params.name_free) {
        const auto it_best = res.best_params.find(name);
        if (it_best == res.best_params.end()) {
            continue;
        }
        task.params.structured.free[name] = it_best->second;
        task.params.flat[name] = it_best->second;
    }
}

#endif // CRITERION_PRIOR_HPP
