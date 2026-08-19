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

        // 检查是否显式传入了 n_conf 作为全量格式标记
        typename std::unordered_map<std::string, std::vector<T>>::const_iterator
            it_n_conf = params.find("n_conf");
        const bool has_n_conf = (it_n_conf != params.end() &&
                                 !it_n_conf->second.empty());
        const bool is_full_vector = (
            has_n_conf &&
            static_cast<std::size_t>(it_n_conf->second[0]) == raw_p.size()
        );

        if (is_full_vector) {
            // 全量向量格式: 要求严格单调递增
            for (std::size_t k = 1; k < raw_p.size(); ++k) {
                if (raw_p[k] <= raw_p[k - 1]) {
                    throw std::invalid_argument(
                        "ModelBCH Initialization Error: Full 'p_conf' vector "
                        "must be strictly increasing."
                    );
                }
            }
            p_thresholds = raw_p;
        } else {
            // 对称简记格式 (例如 p_conf = [p1, p2]):
            // 1. 将每个自由参数值约束至 [1e-4, p_resp - 1e-4]
            // 2. 升序排序
            // 3. 展开为 [p1, p2, p_resp, 1 - p2, 1 - p1]
            const T min_bound = static_cast<T>(1e-4);
            const T max_bound = (p_resp > static_cast<T>(2e-4))
                ? (p_resp - static_cast<T>(1e-4))
                : static_cast<T>(0.4999);

            std::vector<T> sorted_half = raw_p;
            for (std::size_t k = 0; k < sorted_half.size(); ++k) {
                sorted_half[k] = std::max(min_bound, std::min(sorted_half[k], max_bound));
            }
            std::sort(sorted_half.begin(), sorted_half.end());

            // 严格防重递增调整
            for (std::size_t k = 1; k < sorted_half.size(); ++k) {
                if (sorted_half[k] <= sorted_half[k - 1]) {
                    sorted_half[k] = sorted_half[k - 1] + static_cast<T>(1e-6);
                    if (sorted_half[k] >= p_resp) {
                        sorted_half[k] = p_resp - static_cast<T>(1e-6);
                    }
                }
            }

            p_thresholds.clear();
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
        }
    } else {
        p_thresholds.push_back(p_resp);
    }

    // 校验最终概率阈值严格单调递增
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
 *                           Probability Computation                          *
 * ========================================================================== */

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

    auto normal_cdf_fn = [&](const T& x) -> T {
        const T z = (x - mu) / (sd * std::sqrt(static_cast<T>(2.0)));
        return static_cast<T>(0.5) * (static_cast<T>(1.0) + std::erf(z));
    };

    const T inf_thresh = static_cast<T>(1e6);
    const T cdf_upper = (upper >= inf_thresh) ? static_cast<T>(1.0) : normal_cdf_fn(upper);
    const T cdf_lower = (lower <= -inf_thresh) ? static_cast<T>(0.0) : normal_cdf_fn(lower);

    const T diff = cdf_upper - cdf_lower;
    return (diff > static_cast<T>(0.0)) ? diff : static_cast<T>(0.0);
}

template <typename T>
std::vector<std::vector<std::vector<T>>>
ModelBCH<T>::compute_probabilities() const {
    const std::size_t n_diffs = d_vec.size();
    const std::size_t n_criteria = p_thresholds.size();
    const std::size_t n_cols = n_criteria + 1;
    const std::size_t n_ratings = n_cols / 2;

    std::vector<std::vector<std::vector<T>>> prob_mat(
        n_diffs,
        std::vector<std::vector<T>>(
            2,
            std::vector<T>(n_cols, static_cast<T>(0.0))
        )
    );

    const T inf_val = static_cast<T>(1e7);

    for (std::size_t d_idx = 0; d_idx < n_diffs; ++d_idx) {
        const std::vector<T>& crits = criteria_matrix[d_idx];
        for (std::size_t stim = 0; stim < 2; ++stim) {
            // S1: 响应 0
            if (n_ratings == 1) {
                prob_mat[d_idx][stim][0] = area(stim, 0, -inf_val, crits[0], d_idx);
            } else {
                prob_mat[d_idx][stim][0] = area(stim, 0, -inf_val, crits[0], d_idx);
                for (std::size_t k = 1; k < n_ratings - 1; ++k) {
                    prob_mat[d_idx][stim][k] =
                        area(stim, 0, crits[k - 1], crits[k], d_idx);
                }
                prob_mat[d_idx][stim][n_ratings - 1] =
                    area(stim, 0, crits[n_ratings - 2], crits[n_ratings - 1], d_idx);
            }

            // S2: 响应 1
            if (n_ratings == 1) {
                prob_mat[d_idx][stim][1] = area(stim, 1, crits[0], inf_val, d_idx);
            } else {
                prob_mat[d_idx][stim][n_ratings] =
                    area(stim, 1, crits[n_ratings - 1], crits[n_ratings], d_idx);
                for (std::size_t k = 1; k < n_ratings - 1; ++k) {
                    prob_mat[d_idx][stim][n_ratings + k] =
                        area(stim, 1, crits[n_ratings + k - 1], crits[n_ratings + k], d_idx);
                }
                prob_mat[d_idx][stim][n_cols - 1] =
                    area(stim, 1, crits[n_criteria - 1], inf_val, d_idx);
            }
        }
    }

    return prob_mat;
}

template <typename T>
MatrixProb<T> ModelBCH<T>::area() const {
    MatrixProb<T> res;
    res.prob_mat = compute_probabilities();

    const std::size_t n_diffs = res.prob_mat.size();
    const std::size_t n_cols = res.prob_mat[0][0].size();
    const std::size_t n_conf = n_cols / 2;

    T lapse = static_cast<T>(0.0);
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_lapse = std_params.find("rate_lapse");
    if (it_lapse != std_params.end() && !it_lapse->second.empty()) {
        lapse = it_lapse->second[0];
    }

    if (lapse > static_cast<T>(0.0)) {
        for (std::size_t d = 0; d < n_diffs; ++d) {
            for (std::size_t stim = 0; stim < 2; ++stim) {
                for (std::size_t j = 0; j < n_cols; ++j) {
                    res.prob_mat[d][stim][j] =
                        (lapse / static_cast<T>(n_cols)) +
                        ((static_cast<T>(1.0) - lapse) *
                         res.prob_mat[d][stim][j]);
                }
            }
        }
    }

    res.row_names = {"stim_0", "stim_1"};
    if (n_cols == 2) {
        res.col_names = {"resp_0", "resp_1"};
    } else {
        res.col_names.clear();
        for (std::size_t c = n_conf; c > 0; --c) {
            res.col_names.push_back("resp_0_conf_" + std::to_string(c));
        }
        for (std::size_t c = 1; c <= n_conf; ++c) {
            res.col_names.push_back("resp_1_conf_" + std::to_string(c));
        }
    }

    return res;
}

template <typename T>
std::vector<std::vector<T>> ModelBCH<T>::cdf_noise() const {
    std::vector<std::vector<T>> res(d_vec.size());
    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        res[i].resize(criteria_matrix[i].size());
        for (std::size_t k = 0; k < criteria_matrix[i].size(); ++k) {
            res[i][k] = area(0, 0, -static_cast<T>(1e7), criteria_matrix[i][k], i);
        }
    }
    return res;
}

template <typename T>
std::vector<std::vector<T>> ModelBCH<T>::cdf_signal() const {
    std::vector<std::vector<T>> res(d_vec.size());
    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        res[i].resize(criteria_matrix[i].size());
        for (std::size_t k = 0; k < criteria_matrix[i].size(); ++k) {
            res[i][k] = area(1, 0, -static_cast<T>(1e7), criteria_matrix[i][k], i);
        }
    }
    return res;
}

template class ModelBCH<double>;
