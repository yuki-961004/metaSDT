#include <metaSDT/model_bch.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

/* ========================================================================== *
 *                         ModelBCH Construction                              *
 * ========================================================================== */

template <typename T>
ModelBCH<T>::ModelBCH(
    const std::unordered_map<std::string, std::vector<T>>& std_params
) {
    try {
        d_vec = std_params.at("d");
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(
            "ModelBCH Initialization Error: Missing required parameter 'd'."
        );
    }

    if (d_vec.empty()) {
        throw std::invalid_argument(
            "ModelBCH Initialization Error: 'd' vector cannot be empty."
        );
    }

    sd_noise = static_cast<T>(1.0);
    sd_signal = static_cast<T>(1.0);
    if (std_params.count("sd_noise") && !std_params.at("sd_noise").empty()) {
        sd_noise = std_params.at("sd_noise")[0];
    }
    if (std_params.count("sd_signal") && !std_params.at("sd_signal").empty()) {
        sd_signal = std_params.at("sd_signal")[0];
    }

    T p_resp_val = static_cast<T>(0.5);
    if (std_params.count("p_resp") && !std_params.at("p_resp").empty()) {
        p_resp_val = std_params.at("p_resp")[0];
    }

    if (p_resp_val <= static_cast<T>(0.0) || p_resp_val >= static_cast<T>(1.0)) {
        throw std::invalid_argument(
            "ModelBCH Initialization Error: 'p_resp' must be strictly in (0, 1)."
        );
    }

    if (std_params.count("p_conf") && !std_params.at("p_conf").empty()) {
        std::vector<T> p_conf = std_params.at("p_conf");
        std::sort(p_conf.begin(), p_conf.end());

        for (const T& p : p_conf) {
            if (p <= static_cast<T>(0.0) || p >= static_cast<T>(1.0)) {
                throw std::invalid_argument(
                    "ModelBCH Initialization Error: All 'p_conf' values must be strictly in (0, 1)."
                );
            }
        }

        const bool has_n_conf = (std_params.count("n_conf") &&
                                 !std_params.at("n_conf").empty());
        const bool is_full_vector = (
            has_n_conf &&
            static_cast<int>(std_params.at("n_conf")[0]) ==
            static_cast<int>(p_conf.size())
        );

        if (is_full_vector) {
            p_thresholds = p_conf;
        } else {
            p_thresholds.reserve(1 + p_conf.size() * 2);
            for (auto it = p_conf.begin(); it != p_conf.end(); ++it) {
                p_thresholds.push_back(*it);
            }
            p_thresholds.push_back(p_resp_val);
            for (auto it = p_conf.begin(); it != p_conf.end(); ++it) {
                const T mirror = static_cast<T>(1.0) - *it;
                p_thresholds.push_back(mirror);
            }
            std::sort(p_thresholds.begin(), p_thresholds.end());
        }
    } else {
        p_thresholds.push_back(p_resp_val);
    }

    using std::log;
    criteria_matrix.resize(d_vec.size());
    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        const T d_val = d_vec[i];
        if (d_val <= static_cast<T>(0.0)) {
            throw std::invalid_argument(
                "ModelBCH Initialization Error: 'd' values must be strictly positive."
            );
        }
        criteria_matrix[i].resize(p_thresholds.size());
        for (std::size_t k = 0; k < p_thresholds.size(); ++k) {
            const T p = p_thresholds[k];
            const T logit_p = log(p / (static_cast<T>(1.0) - p));
            criteria_matrix[i][k] = logit_p / d_val;
        }
    }
}

/* ========================================================================== *
 *                           CDF Matrix Builders                              *
 * ========================================================================== */

template <typename T>
std::vector<std::vector<T>> ModelBCH<T>::cdf_noise() const {
    using std::erf;
    using std::sqrt;

    std::vector<std::vector<T>> res(d_vec.size());
    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        const T mu_noise = -d_vec[i] * static_cast<T>(0.5);
        res[i].resize(criteria_matrix[i].size());
        for (std::size_t k = 0; k < criteria_matrix[i].size(); ++k) {
            const T z = (criteria_matrix[i][k] - mu_noise) /
                        (sd_noise * sqrt(static_cast<T>(2.0)));
            res[i][k] = static_cast<T>(0.5) * (static_cast<T>(1.0) + erf(z));
        }
    }
    return res;
}

template <typename T>
std::vector<std::vector<T>> ModelBCH<T>::cdf_signal() const {
    using std::erf;
    using std::sqrt;

    std::vector<std::vector<T>> res(d_vec.size());
    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        const T mu_signal = d_vec[i] * static_cast<T>(0.5);
        res[i].resize(criteria_matrix[i].size());
        for (std::size_t k = 0; k < criteria_matrix[i].size(); ++k) {
            const T z = (criteria_matrix[i][k] - mu_signal) /
                        (sd_signal * sqrt(static_cast<T>(2.0)));
            res[i][k] = static_cast<T>(0.5) * (static_cast<T>(1.0) + erf(z));
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
    using std::erf;
    using std::sqrt;

    (void)response;
    const T mu = (stimulus == 0)
        ? (-d_vec[dim_idx] * static_cast<T>(0.5))
        : (d_vec[dim_idx] * static_cast<T>(0.5));
    const T sd = (stimulus == 0) ? sd_noise : sd_signal;

    auto normal_cdf = [&](const T& x) -> T {
        const T z = (x - mu) / (sd * sqrt(static_cast<T>(2.0)));
        return static_cast<T>(0.5) * (static_cast<T>(1.0) + erf(z));
    };

    return normal_cdf(upper) - normal_cdf(lower);
}

template class ModelBCH<double>;
