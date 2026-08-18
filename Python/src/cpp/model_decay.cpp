#include <metaSDT/model_decay.hpp>
#include <metaSDT/matrix_prob.hpp>
#include <metaSDT/quadrature.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/* ========================================================================== *
 *                           ModelDecay Construction                          *
 * ========================================================================== */

template <typename T>
ModelDecay<T>::ModelDecay(
    const std::unordered_map<std::string, std::vector<T>>& params
) : std_params(params) {
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_d = params.find("d");
    if (it_d == params.end() || it_d->second.empty()) {
        throw std::invalid_argument(
            "ModelDecay Initialization Error: Missing parameter 'd'."
        );
    }
    d_vec = it_d->second;

    for (std::size_t i = 0; i < d_vec.size(); ++i) {
        if (d_vec[i] <= static_cast<T>(0.0)) {
            throw std::invalid_argument(
                "ModelDecay Initialization Error: 'd' values must be positive."
            );
        }
    }

    c_resp = static_cast<T>(0.0);
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_c_resp = params.find("c_resp");
    if (it_c_resp != params.end() && !it_c_resp->second.empty()) {
        c_resp = it_c_resp->second[0];
    }

    sigma_meta = static_cast<T>(0.0);
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_sigma_meta = params.find("sigma_meta");
    if (it_sigma_meta != params.end() && !it_sigma_meta->second.empty()) {
        sigma_meta = it_sigma_meta->second[0];
    }
    if (sigma_meta < static_cast<T>(0.0)) {
        throw std::invalid_argument(
            "ModelDecay Initialization Error: 'sigma_meta' must be "
            "non-negative."
        );
    }

    // 处理逐难度的衰减参数 rho_decay (delta)
    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_rho = params.find("rho_decay");
    if (it_rho != params.end() && !it_rho->second.empty()) {
        const std::vector<T>& raw_rho = it_rho->second;
        rho_decay = raw_rho;
        if (rho_decay.size() < d_vec.size()) {
            rho_decay.resize(d_vec.size(), raw_rho[0]);
        }
    } else {
        rho_decay.assign(d_vec.size(), static_cast<T>(0.99));
    }

    for (std::size_t i = 0; i < rho_decay.size(); ++i) {
        const T val = rho_decay[i];
        if (val <= static_cast<T>(0.0) || val > static_cast<T>(1.0)) {
            throw std::invalid_argument(
                "ModelDecay Initialization Error: 'rho_decay' values must be "
                "in (0, 1]."
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
            "ModelDecay Initialization Error: ModelDecay requires equal "
            "unit variance (sd_noise = 1.0, sd_signal = 1.0)."
        );
    }

    typename std::unordered_map<std::string, std::vector<T>>::const_iterator
        it_c_conf = params.find("c_conf");
    if (it_c_conf != params.end() && !it_c_conf->second.empty()) {
        std::vector<T> raw_conf = it_c_conf->second;
        std::sort(raw_conf.begin(), raw_conf.end());

        typename std::unordered_map<std::string, std::vector<T>>::const_iterator
            it_n_conf = params.find("n_conf");
        const bool has_n_conf = (it_n_conf != params.end() &&
                                 !it_n_conf->second.empty());
        const bool is_full_vector = (
            has_n_conf &&
            static_cast<std::size_t>(it_n_conf->second[0]) == raw_conf.size()
        );

        if (is_full_vector) {
            criteria = raw_conf;
        } else {
            criteria.reserve(1 + raw_conf.size() * 2);
            for (auto it = raw_conf.rbegin(); it != raw_conf.rend(); ++it) {
                criteria.push_back(c_resp - *it);
            }
            criteria.push_back(c_resp);
            for (auto it = raw_conf.begin(); it != raw_conf.end(); ++it) {
                criteria.push_back(c_resp + *it);
            }
        }
        c_conf = raw_conf;
    } else {
        criteria.push_back(c_resp);
    }
}

/* ========================================================================== *
 *                         Probability Computation                            *
 * ========================================================================== */

namespace {

template <typename T>
T normal_pdf(const T& x, const T& mu, const T& sigma) {
    const T z = (x - mu) / sigma;
    const T inv_sqrt_2pi = static_cast<T>(0.39894228040143267794);
    return (inv_sqrt_2pi / sigma) * std::exp(static_cast<T>(-0.5) * z * z);
}

template <typename T>
T normal_cdf(const T& x, const T& mu, const T& sigma) {
    const T z = (x - mu) / (sigma * std::sqrt(static_cast<T>(2.0)));
    return static_cast<T>(0.5) * (static_cast<T>(1.0) + std::erf(z));
}

template <typename T>
T prob_high_conf_decay(
    const T& mu1,
    const T& c,
    const T& conf_crit,
    const T& sigma_meta,
    const T& delta
) {
    if (sigma_meta <= static_cast<T>(1e-6)) {
        const T eff_cut = std::max(c, conf_crit / delta);
        return static_cast<T>(1.0) -
               normal_cdf(eff_cut, mu1, static_cast<T>(1.0));
    }

    const T b = std::max(c, mu1) + static_cast<T>(7.0);
    if (b <= c) {
        return static_cast<T>(0.0);
    }

    auto integrand = [&](const T& x) -> T {
        const T pdf_val = normal_pdf(x, mu1, static_cast<T>(1.0));
        const T cdf_val = normal_cdf(delta * x, conf_crit, sigma_meta);
        return pdf_val * cdf_val;
    };

    return Quadrature::integrate_32(integrand, c, b);
}

} // namespace

template <typename T>
std::vector<std::vector<std::vector<T>>>
ModelDecay<T>::compute_probabilities() const {
    const std::size_t n_diffs = d_vec.size();
    const std::size_t n_criteria = criteria.size();
    const std::size_t n_cols = n_criteria + 1;
    const std::size_t n_ratings = n_cols / 2;

    std::vector<std::vector<std::vector<T>>> prob_mat(
        n_diffs,
        std::vector<std::vector<T>>(
            2,
            std::vector<T>(n_cols, static_cast<T>(0.0))
        )
    );

    const std::size_t mid_idx = n_ratings - 1;

    for (std::size_t d_idx = 0; d_idx < n_diffs; ++d_idx) {
        const T d = d_vec[d_idx];
        const T delta = rho_decay[d_idx];

        for (std::size_t stim = 0; stim < 2; ++stim) {
            const T mu_stim = (stim == 0)
                ? (-d * static_cast<T>(0.5))
                : (d * static_cast<T>(0.5));

            // S1 (resp = 0)
            const T q_neg = normal_cdf(c_resp, mu_stim, sd_noise);
            std::vector<T> p_neg(n_ratings - 1);
            for (std::size_t k = 0; k < n_ratings - 1; ++k) {
                const T crit_val = -criteria[k];
                p_neg[k] = prob_high_conf_decay(
                    -mu_stim,
                    -c_resp,
                    crit_val,
                    sigma_meta,
                    delta
                );
            }

            if (n_ratings == 1) {
                prob_mat[d_idx][stim][0] = q_neg;
            } else {
                prob_mat[d_idx][stim][0] = p_neg[0];
                for (std::size_t k = 1; k < n_ratings - 1; ++k) {
                    prob_mat[d_idx][stim][k] = p_neg[k] - p_neg[k - 1];
                }
                prob_mat[d_idx][stim][n_ratings - 1] =
                    q_neg - p_neg[n_ratings - 2];
            }

            // S2 (resp = 1)
            const T q_pos = static_cast<T>(1.0) -
                            normal_cdf(c_resp, mu_stim, sd_signal);
            std::vector<T> p_pos(n_ratings - 1);
            for (std::size_t k = 0; k < n_ratings - 1; ++k) {
                const T crit_val = criteria[mid_idx + 1 + k];
                p_pos[k] = prob_high_conf_decay(
                    mu_stim,
                    c_resp,
                    crit_val,
                    sigma_meta,
                    delta
                );
            }

            if (n_ratings == 1) {
                prob_mat[d_idx][stim][1] = q_pos;
            } else {
                prob_mat[d_idx][stim][n_ratings] = q_pos - p_pos[0];
                for (std::size_t k = 1; k < n_ratings - 1; ++k) {
                    prob_mat[d_idx][stim][n_ratings + k] =
                        p_pos[k - 1] - p_pos[k];
                }
                prob_mat[d_idx][stim][n_cols - 1] = p_pos[n_ratings - 2];
            }
        }
    }

    return prob_mat;
}

template <typename T>
T ModelDecay<T>::area(
    std::size_t stimulus,
    std::size_t response,
    const T& lower,
    const T& upper,
    std::size_t dim_idx
) const {
    const T d = d_vec[dim_idx];
    const T delta = rho_decay[dim_idx];
    const T mu_stim = (stimulus == 0)
        ? (-d * static_cast<T>(0.5))
        : (d * static_cast<T>(0.5));

    if (response == 1) {
        const T p_lower = prob_high_conf_decay(
            mu_stim,
            c_resp,
            lower,
            sigma_meta,
            delta
        );
        const T p_upper = prob_high_conf_decay(
            mu_stim,
            c_resp,
            upper,
            sigma_meta,
            delta
        );
        return p_lower - p_upper;
    } else {
        const T p_lower = prob_high_conf_decay(
            -mu_stim,
            -c_resp,
            -lower,
            sigma_meta,
            delta
        );
        const T p_upper = prob_high_conf_decay(
            -mu_stim,
            -c_resp,
            -upper,
            sigma_meta,
            delta
        );
        return p_upper - p_lower;
    }
}

template <typename T>
MatrixProb<T> ModelDecay<T>::area() const {
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

template class ModelDecay<double>;
