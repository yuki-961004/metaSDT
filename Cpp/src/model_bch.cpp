#include <metaSDT/model_bch.hpp>
#include <metaSDT/matrix_prob.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/* ========================================================================== *
 *                         ModelBCH Construction                              *
 * ========================================================================== */

template <typename T>
ModelBCH<T>::ModelBCH(
    const std::unordered_map<std::string, std::vector<T>>& params
) : std_params(params) {
    // 提取敏感度向量 d
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_d = params.find("d");
    if (it_d == params.end() || it_d->second.empty()) {
        throw std::invalid_argument(
            "ModelBCH Initialization Error: Missing required parameter 'd'."
        );
    }
    d_vec = it_d->second;

    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        if (d_vec[i] <= static_cast<T>(0.0)) {
            throw std::invalid_argument(
                "ModelBCH Initialization Error: 'd' values must be positive."
            );
        }
    }

    // 第一阶段实施严格等方差政策 (sd_noise = 1.0, sd_signal = 1.0)
    sd_noise = static_cast<T>(1.0);
    sd_signal = static_cast<T>(1.0);
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_sd_noise = params.find("sd_noise");
    if (it_sd_noise != params.end() && !it_sd_noise->second.empty()) {
        sd_noise = it_sd_noise->second[0];
    }
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_sd_signal = params.find("sd_signal");
    if (it_sd_signal != params.end() && !it_sd_signal->second.empty()) {
        sd_signal = it_sd_signal->second[0];
    }

    if (sd_noise != static_cast<T>(1.0) || sd_signal != static_cast<T>(1.0)) {
        throw std::invalid_argument(
            "ModelBCH Initialization Error: ModelBCH requires equal unit "
            "variance (sd_noise = 1.0, sd_signal = 1.0)."
        );
    }

    // 提取决策边界对应的先验/后验概率阈值 p_resp (默认 0.5)
    p_resp = static_cast<T>(0.5);
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_p_resp = params.find("p_resp");
    if (it_p_resp != params.end() && !it_p_resp->second.empty()) {
        p_resp = it_p_resp->second[0];
    }

    if (p_resp <= static_cast<T>(0.0) || p_resp >= static_cast<T>(1.0)) {
        throw std::invalid_argument(
            "ModelBCH Initialization Error: 'p_resp' must be in (0, 1)."
        );
    }

    // 处理置信度概率阈值 p_conf
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_p_conf = params.find("p_conf");
    if (it_p_conf != params.end() && !it_p_conf->second.empty()) {
        const std::vector<T>& raw_p = it_p_conf->second;
        for (std::size_t i = 0; i < raw_p.size(); ++i) {
            if (raw_p[i] <= static_cast<T>(0.0) ||
                raw_p[i] >= static_cast<T>(1.0)) {
                throw std::invalid_argument(
                    "ModelBCH Initialization Error: 'p_conf' values must be "
                    "in (0, 1)."
                );
            }
        }

        // 检查是否显式传入了 n_conf
        typename std::unordered_map<std::string, std::vector<T>>::const_iterator
            it_n_conf = params.find("n_conf");
        const bool has_n_conf = (it_n_conf != params.end() &&
                                 !it_n_conf->second.empty());

        if (has_n_conf &&
            static_cast<std::size_t>(it_n_conf->second[0]) == raw_p.size()) {
            p_thresholds = raw_p;
        } else {
            // 判断是否为跨越 p_resp 的全量非对称阈值
            bool all_less = true;
            bool is_asymmetric_pair = false;
            for (std::size_t k = 0; k < raw_p.size(); ++k) {
                if (raw_p[k] >= p_resp) {
                    all_less = false;
                    break;
                }
            }

            if (raw_p.size() % 2 == 0 && raw_p.size() >= 2 && !all_less) {
                const std::size_t half = raw_p.size() / 2;
                bool valid_split = true;
                for (std::size_t k = 0; k < half; ++k) {
                    if (raw_p[k] >= p_resp) {
                        valid_split = false;
                        break;
                    }
                }
                for (std::size_t k = half; k < raw_p.size(); ++k) {
                    if (raw_p[k] <= p_resp) {
                        valid_split = false;
                        break;
                    }
                }
                if (valid_split) {
                    is_asymmetric_pair = true;
                }
            }

            if (is_asymmetric_pair) {
                // MATLAB 原版格式: 包含前半段 (< p_resp) 与后半段 (> p_resp)
                p_thresholds.reserve(raw_p.size() + 1);
                const std::size_t half = raw_p.size() / 2;
                for (std::size_t k = 0; k < half; ++k) {
                    p_thresholds.push_back(raw_p[k]);
                }
                p_thresholds.push_back(p_resp);
                for (std::size_t k = half; k < raw_p.size(); ++k) {
                    p_thresholds.push_back(raw_p[k]);
                }
            } else if (all_less) {
                // 对称简记形式: 传入单侧阈值, 围绕 p_resp 镜像展开
                std::vector<T> sorted_half = raw_p;
                std::sort(sorted_half.begin(), sorted_half.end());
                p_thresholds.reserve(sorted_half.size() * 2 + 1);
                for (std::size_t k = 0; k < sorted_half.size(); ++k) {
                    p_thresholds.push_back(sorted_half[k]);
                }
                p_thresholds.push_back(p_resp);
                for (std::size_t k = 0; k < sorted_half.size(); ++k) {
                    p_thresholds.push_back(
                        static_cast<T>(1.0) -
                        sorted_half[sorted_half.size() - 1 - k]
                    );
                }
            } else {
                p_thresholds = raw_p;
                std::sort(p_thresholds.begin(), p_thresholds.end());
            }
        }
    } else {
        p_thresholds.push_back(p_resp);
    }

    // 校验概率阈值严格单调递增
    for (std::size_t k = 1; k < p_thresholds.size(); ++k) {
        if (p_thresholds[k] <= p_thresholds[k - 1]) {
            throw std::invalid_argument(
                "ModelBCH Initialization Error: Probability thresholds must "
                "be strictly increasing."
            );
        }
    }

    // 将概率空间阈值转换至证据空间: c_k = log(p_k / (1 - p_k)) / d
    criteria_matrix.resize(d_vec.size());
    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        const T d_val = d_vec[i];
        criteria_matrix[i].resize(p_thresholds.size());
        for (std::size_t k = 0; k < p_thresholds.size(); ++k) {
            const T p = p_thresholds[k];
            const T logit_p = std::log(p / (static_cast<T>(1.0) - p));
            criteria_matrix[i][k] = logit_p / d_val;
        }
    }
}

/* ========================================================================== *
 *                           CDF Matrix Builders                              *
 * ========================================================================== */

template <typename T>
std::vector<std::vector<T>> ModelBCH<T>::cdf_noise() const {
    std::vector<std::vector<T>> res(d_vec.size());
    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        const T mu_noise = -d_vec[i] * static_cast<T>(0.5);
        res[i].resize(criteria_matrix[i].size());
        for (std::size_t k = 0; k < criteria_matrix[i].size(); ++k) {
            const T z = (criteria_matrix[i][k] - mu_noise) /
                        (sd_noise * std::sqrt(static_cast<T>(2.0)));
            res[i][k] = static_cast<T>(0.5) *
                        (static_cast<T>(1.0) + std::erf(z));
        }
    }
    return res;
}

template <typename T>
std::vector<std::vector<T>> ModelBCH<T>::cdf_signal() const {
    std::vector<std::vector<T>> res(d_vec.size());
    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        const T mu_signal = d_vec[i] * static_cast<T>(0.5);
        res[i].resize(criteria_matrix[i].size());
        for (std::size_t k = 0; k < criteria_matrix[i].size(); ++k) {
            const T z = (criteria_matrix[i][k] - mu_signal) /
                        (sd_signal * std::sqrt(static_cast<T>(2.0)));
            res[i][k] = static_cast<T>(0.5) *
                        (static_cast<T>(1.0) + std::erf(z));
        }
    }
    return res;
}

template <typename T>
T ModelBCH<T>::area(
    std::size_t stimulus,
    std::size_t response,
    const T& lower,
    const T& upper,
    std::size_t dim_idx
) const {
    (void)response;
    const T mu = (stimulus == 0)
        ? (-d_vec[dim_idx] * static_cast<T>(0.5))
        : (d_vec[dim_idx] * static_cast<T>(0.5));
    const T sd = (stimulus == 0) ? sd_noise : sd_signal;

    auto normal_cdf = [&](const T& x) -> T {
        const T z = (x - mu) / (sd * std::sqrt(static_cast<T>(2.0)));
        return static_cast<T>(0.5) * (static_cast<T>(1.0) + std::erf(z));
    };

    return normal_cdf(upper) - normal_cdf(lower);
}

template <typename T>
MatrixProb<T> ModelBCH<T>::area() const {
    return matrix_prob<T>(cdf_noise(), cdf_signal(), std_params);
}

template class ModelBCH<double>;
