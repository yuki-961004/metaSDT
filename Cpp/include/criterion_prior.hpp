#ifndef CRITERION_PRIOR_HPP
#define CRITERION_PRIOR_HPP

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

class CriterionPrior {
public:
    enum class PriorType {
        NORMAL,
        UNIFORM,
        LOGNORMAL,
        CAUCHY,
        BETA,
        EXPONENTIAL,
        NONE
    };

    struct PriorSpec {
        PriorType type;
        double param1;
        double param2;
    };

    std::unordered_map<int, PriorSpec> prior_specs_;

    void add_prior(int param_index, PriorType type, double p1, double p2) {
        prior_specs_[param_index] = {type, p1, p2};
    }

    template <typename T>
    T evaluate(const Eigen::Matrix<T, Eigen::Dynamic, 1>& free_params) const {
        T log_prior = 0.0;

        using std::log;
        using std::pow;
        const double PI = 3.14159265358979323846;

        for (const auto& kv : prior_specs_) {
            const int idx = kv.first;
            const auto& spec = kv.second;

            if (idx >= free_params.size()) {
                continue;
            }
            const T val = free_params(idx);

            switch (spec.type) {
                case PriorType::NORMAL: {
                    const double mu = spec.param1;
                    const double sigma = spec.param2;
                    log_prior += -0.5 * log(2.0 * PI * sigma * sigma)
                                 - 0.5 * pow((val - mu) / sigma, 2);
                    break;
                }
                case PriorType::LOGNORMAL: {
                    const double mu = spec.param1;
                    const double sigma = spec.param2;
                    if (val > 0.0) {
                        log_prior += -log(val * sigma * sqrt(2.0 * PI))
                                     - 0.5 * pow((log(val) - mu) / sigma, 2);
                    } else {
                        log_prior += -1e10;
                    }
                    break;
                }
                case PriorType::CAUCHY: {
                    const double location = spec.param1;
                    const double scale = spec.param2;
                    log_prior += -log(PI * scale)
                                 - log(1.0 + pow((val - location) / scale, 2));
                    break;
                }
                case PriorType::UNIFORM: {
                    const double min_val = spec.param1;
                    const double max_val = spec.param2;
                    if (val >= min_val && val <= max_val) {
                        log_prior += -log(max_val - min_val);
                    } else {
                        log_prior += -1e10;
                    }
                    break;
                }
                case PriorType::BETA: {
                    const double shape1 = spec.param1;
                    const double shape2 = spec.param2;
                    if (val > 0.0 && val < 1.0) {
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

    void update(const std::vector<std::vector<double>>& group_free_params) {
        if (group_free_params.empty()) {
            return;
        }
        const size_t n_subjects = group_free_params.size();
        if (n_subjects < 2) {
            return;
        }

        const size_t k_params = group_free_params[0].size();

        for (auto& kv : prior_specs_) {
            const int j = kv.first;
            if (j >= static_cast<int>(k_params)) {
                continue;
            }

            PriorSpec& spec = kv.second;
            if (spec.type == PriorType::NONE ||
                spec.type == PriorType::UNIFORM) {
                continue;
            }

            std::vector<double> x(n_subjects);
            for (size_t i = 0; i < n_subjects; ++i) {
                x[i] = group_free_params[i][j];
            }

            const double mean =
                std::accumulate(x.begin(), x.end(), 0.0) / n_subjects;
            double var = 0.0;
            for (const double v : x) {
                var += (v - mean) * (v - mean);
            }
            var /= (n_subjects - 1.0);

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
