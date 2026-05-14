#include "../include/model_sdt.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <typename T>
ModelSDT<T>::ModelSDT(
    const std::unordered_map<std::string, std::vector<T>>& std_params
) {
    try {
        d_vec = std_params.at("d");
        sd_noise = std_params.at("sd_noise")[0];
        sd_signal = std_params.at("sd_signal")[0];
    } catch (const std::out_of_range& e) {
        throw std::invalid_argument(
            "ModelSDT Initialization Error: Missing required parameters "
            "('d', 'sd_noise', or 'sd_signal')."
        );
    }

    T sort_d = 0.0;
    if (std_params.count("sort_d") && !std_params.at("sort_d").empty()) {
        sort_d = std_params.at("sort_d")[0];
    }
    if (sort_d != 0.0) {
        std::sort(d_vec.rbegin(), d_vec.rend());
    }

    for (size_t i = 0; i < d_vec.size(); ++i) {
        mu_noise_vec.push_back(-d_vec[i] / 2.0);
        mu_signal_vec.push_back(d_vec[i] / 2.0);
    }

    T c_resp_val = 0.0;
    auto it_c_resp = std_params.find("c_resp");
    if (it_c_resp != std_params.end() && !it_c_resp->second.empty()) {
        c_resp_val = it_c_resp->second[0];
    }

    auto it_c_conf = std_params.find("c_conf");
    if (it_c_conf != std_params.end() && !it_c_conf->second.empty()) {
        std::vector<T> c_conf = it_c_conf->second;
        std::sort(c_conf.begin(), c_conf.end());

        auto it_n_conf = std_params.find("n_conf");
        bool has_n_conf = (it_n_conf != std_params.end() &&
                           !it_n_conf->second.empty());
        bool is_full_vector = (
            has_n_conf &&
            static_cast<int>(it_n_conf->second[0]) ==
            static_cast<int>(c_conf.size())
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

template <typename T>
T ModelSDT<T>::cdf_noise(T x, size_t dim_idx) const {
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

template <typename T>
std::vector<T> ModelSDT<T>::cdf_noise(
    const std::vector<T>& x_vec,
    size_t dim_idx
) const {
    std::vector<T> y_vec(x_vec.size());
    for (size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_noise(/*x=*/x_vec[i], /*dim_idx=*/dim_idx);
    }
    return y_vec;
}

template <typename T>
std::vector<T> ModelSDT<T>::cdf_signal(
    const std::vector<T>& x_vec,
    size_t dim_idx
) const {
    std::vector<T> y_vec(x_vec.size());
    for (size_t i = 0; i < x_vec.size(); ++i) {
        y_vec[i] = cdf_signal(/*x=*/x_vec[i], /*dim_idx=*/dim_idx);
    }
    return y_vec;
}

template class ModelSDT<double>;
