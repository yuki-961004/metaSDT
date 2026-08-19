#include <metaSDT/model_sdt.hpp>
#include <metaSDT/matrix_prob.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================== *
 *                         ModelSDT Construction                              *
 * ========================================================================== */

template <typename T>
ModelSDT<T>::ModelSDT(
    const std::unordered_map<std::string, std::vector<T>>& params
) : std_params(params) {
    try {
        d_vec = params.at("d");
        sd_noise = params.at("sd_noise")[0];
        sd_signal = params.at("sd_signal")[0];
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(
            "ModelSDT Initialization Error: Missing required parameters "
            "('d', 'sd_noise', or 'sd_signal')."
        );
    }

    T sort_d = static_cast<T>(0.0);
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_sort = params.find("sort_d");
    if (it_sort != params.end() && !it_sort->second.empty()) {
        sort_d = it_sort->second[0];
    }

    if (sort_d != static_cast<T>(0.0)) {
        std::sort(d_vec.rbegin(), d_vec.rend());
    }

    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        mu_noise_vec.push_back(-d_vec[i] * static_cast<T>(0.5));
        mu_signal_vec.push_back(d_vec[i] * static_cast<T>(0.5));
    }

    T c_resp_val = static_cast<T>(0.0);
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_c_resp = params.find("c_resp");
    if (it_c_resp != params.end() && !it_c_resp->second.empty()) {
        c_resp_val = it_c_resp->second[0];
    }

    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_c_conf = params.find("c_conf");
    if (it_c_conf != params.end() && !it_c_conf->second.empty()) {
        std::vector<T> c_conf = it_c_conf->second;
        std::sort(c_conf.begin(), c_conf.end());

        typename std::unordered_map<std::string, std::vector<T>>::const_iterator
            it_n_conf = params.find("n_conf");
        const bool has_n_conf = (it_n_conf != params.end() &&
                                 !it_n_conf->second.empty());
        const bool is_full_vector = (
            has_n_conf &&
            static_cast<std::size_t>(it_n_conf->second[0]) == c_conf.size()
        );

        if (is_full_vector) {
            criteria = c_conf;
        } else {
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
        criteria.push_back(c_resp_val);
    }
}

/* ========================================================================== *
 *                           CDF Matrix Builders                              *
 * ========================================================================== */

template <typename T>
std::vector<std::vector<T>> ModelSDT<T>::cdf_noise() const {
    std::vector<std::vector<T>> res(d_vec.size());
    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        res[i] = cdf_noise(this->criteria, i);
    }
    return res;
}

template <typename T>
std::vector<std::vector<T>> ModelSDT<T>::cdf_signal() const {
    std::vector<std::vector<T>> res(d_vec.size());
    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        res[i] = cdf_signal(this->criteria, i);
    }
    return res;
}

template <typename T>
T ModelSDT<T>::cdf_noise(T x, std::size_t dim_idx) const {
    return static_cast<T>(0.5) * (static_cast<T>(1.0) + std::erf(
        (x - mu_noise_vec[dim_idx]) /
        (sd_noise * std::sqrt(static_cast<T>(2.0)))
    ));
}

template <typename T>
T ModelSDT<T>::cdf_signal(T x, std::size_t dim_idx) const {
    return static_cast<T>(0.5) * (static_cast<T>(1.0) + std::erf(
        (x - mu_signal_vec[dim_idx]) /
        (sd_signal * std::sqrt(static_cast<T>(2.0)))
    ));
}

template <typename T>
std::vector<T> ModelSDT<T>::cdf_noise(
    const std::vector<T>& x_vec,
    std::size_t dim_idx
) const {
    std::vector<T> y_vec(x_vec.size());
    for (std::size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_noise(x_vec[i], dim_idx);
    }
    return y_vec;
}

template <typename T>
std::vector<T> ModelSDT<T>::cdf_signal(
    const std::vector<T>& x_vec,
    std::size_t dim_idx
) const {
    std::vector<T> y_vec(x_vec.size());
    for (std::size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_signal(x_vec[i], dim_idx);
    }
    return y_vec;
}

template <typename T>
T ModelSDT<T>::area(
    std::size_t stimulus,
    std::size_t response,
    const T& lower,
    const T& upper,
    std::size_t dim_idx
) const {
    (void)response;
    const T inf_thresh = static_cast<T>(1e6);
    const T cdf_upper = (upper >= inf_thresh)
        ? static_cast<T>(1.0)
        : ((stimulus == 0) ? cdf_noise(upper, dim_idx) : cdf_signal(upper, dim_idx));
    const T cdf_lower = (lower <= -inf_thresh)
        ? static_cast<T>(0.0)
        : ((stimulus == 0) ? cdf_noise(lower, dim_idx) : cdf_signal(lower, dim_idx));

    const T diff = cdf_upper - cdf_lower;
    return (diff > static_cast<T>(0.0)) ? diff : static_cast<T>(0.0);
}

template <typename T>
MatrixProb<T> ModelSDT<T>::area() const {
    return matrix_prob<T>(cdf_noise(), cdf_signal(), std_params);
}

template class ModelSDT<double>;
