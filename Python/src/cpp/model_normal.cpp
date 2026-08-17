#include <metaSDT/model_normal.hpp>
#include <metaSDT/quadrature.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================== *
 *                         ModelNormal Construction                           *
 * ========================================================================== */

template <typename T>
ModelNormal<T>::ModelNormal(
    const std::unordered_map<std::string, std::vector<T>>& std_params
) {
    try {
        d_vec = std_params.at("d");
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(
            "ModelNormal Initialization Error: Missing required parameter 'd'."
        );
    }

    if (d_vec.empty()) {
        throw std::invalid_argument(
            "ModelNormal Initialization Error: 'd' vector cannot be empty."
        );
    }

    c_resp = static_cast<T>(0.0);
    if (std_params.count("c_resp") && !std_params.at("c_resp").empty()) {
        c_resp = std_params.at("c_resp")[0];
    }

    sigma_meta = static_cast<T>(0.5);
    if (std_params.count("sigma_meta") && !std_params.at("sigma_meta").empty()) {
        sigma_meta = std_params.at("sigma_meta")[0];
    }

    if (sigma_meta <= static_cast<T>(0.0)) {
        throw std::invalid_argument(
            "ModelNormal Initialization Error: 'sigma_meta' must be strictly positive."
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

    if (std_params.count("c_conf") && !std_params.at("c_conf").empty()) {
        std::vector<T> raw_conf = std_params.at("c_conf");
        std::sort(raw_conf.begin(), raw_conf.end());

        const bool has_n_conf = (std_params.count("n_conf") &&
                                 !std_params.at("n_conf").empty());
        const bool is_full_vector = (
            has_n_conf &&
            static_cast<int>(std_params.at("n_conf")[0]) ==
            static_cast<int>(raw_conf.size())
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
    using std::exp;
    using std::sqrt;
    const T z = (x - mu) / sigma;
    const T inv_sqrt_2pi = static_cast<T>(0.39894228040143267794);
    return (inv_sqrt_2pi / sigma) * exp(static_cast<T>(-0.5) * z * z);
}

template <typename T>
T normal_cdf(const T& x, const T& mu, const T& sigma) {
    using std::erf;
    using std::sqrt;
    const T z = (x - mu) / (sigma * sqrt(static_cast<T>(2.0)));
    return static_cast<T>(0.5) * (static_cast<T>(1.0) + erf(z));
}

// Numerical evaluation of P(x > c and y > c_conf) with x ~ N(mu1, 1), y|x ~ N(x, sigma_meta^2)
template <typename T>
T prob_high_conf_normal(const T& mu1, const T& c, const T& conf_crit, const T& sigma_meta) {
    using std::max;

    const T b = max(c, mu1) + static_cast<T>(7.0);
    if (b <= c) {
        return static_cast<T>(0.0);
    }

    auto integrand = [&](const T& x) -> T {
        const T pdf_val = normal_pdf(x, mu1, static_cast<T>(1.0));
        const T cdf_val = normal_cdf(x, conf_crit, sigma_meta);
        return pdf_val * cdf_val;
    };

    return Quadrature::integrate_32(integrand, c, b);
}

} // namespace

template <typename T>
std::vector<std::vector<std::vector<T>>> ModelNormal<T>::compute_probabilities() const {
    const std::size_t n_diffs = d_vec.size();
    const std::size_t n_criteria = criteria.size();
    const std::size_t n_cols = n_criteria + 1;
    const std::size_t n_ratings = n_cols / 2;

    std::vector<std::vector<std::vector<T>>> prob_mat(
        n_diffs,
        std::vector<std::vector<T>>(2, std::vector<T>(n_cols, static_cast<T>(0.0)))
    );

    // Extract confidence criteria for left (S1) and right (S2)
    // criteria is [c_1, c_2, ..., c_{K-1}, c_resp, c_{K+1}, ..., c_{2K-1}]
    const std::size_t mid_idx = n_ratings - 1;

    for (std::size_t d_idx = 0; d_idx < n_diffs; ++d_idx) {
        const T d = d_vec[d_idx];

        for (std::size_t stim = 0; stim < 2; ++stim) {
            const T mu_stim = (stim == 0)
                ? (-d * static_cast<T>(0.5))
                : (d * static_cast<T>(0.5));

            // S1 Responses (resp = 0):
            // Type-1 probability for resp 0
            const T q_neg = normal_cdf(c_resp, mu_stim, sd_noise);
            std::vector<T> p_neg(n_ratings - 1);
            for (std::size_t k = 0; k < n_ratings - 1; ++k) {
                // negative criteria flipped relative to -c_resp
                const T crit_val = -criteria[k];
                p_neg[k] = prob_high_conf_normal(-mu_stim, -c_resp, crit_val, sigma_meta);
            }

            // S1 bin probabilities:
            // High confidence (rating K): p_neg[0]
            // Intermediate ratings: p_neg[k] - p_neg[k-1]
            // Rating 1: q_neg - p_neg[n_ratings - 2]
            if (n_ratings == 1) {
                prob_mat[d_idx][stim][0] = q_neg;
            } else {
                prob_mat[d_idx][stim][0] = p_neg[0];
                for (std::size_t k = 1; k < n_ratings - 1; ++k) {
                    prob_mat[d_idx][stim][k] = p_neg[k] - p_neg[k - 1];
                }
                prob_mat[d_idx][stim][n_ratings - 1] = q_neg - p_neg[n_ratings - 2];
            }

            // S2 Responses (resp = 1):
            // Type-1 probability for resp 1
            const T q_pos = static_cast<T>(1.0) - normal_cdf(c_resp, mu_stim, sd_signal);
            std::vector<T> p_pos(n_ratings - 1);
            for (std::size_t k = 0; k < n_ratings - 1; ++k) {
                const T crit_val = criteria[mid_idx + 1 + k];
                p_pos[k] = prob_high_conf_normal(mu_stim, c_resp, crit_val, sigma_meta);
            }

            // S2 bin probabilities:
            // Rating 1: q_pos - p_pos[0]
            // Intermediate ratings: p_pos[k-1] - p_pos[k]
            // High confidence (rating K): p_pos[n_ratings - 2]
            if (n_ratings == 1) {
                prob_mat[d_idx][stim][1] = q_pos;
            } else {
                prob_mat[d_idx][stim][n_ratings] = q_pos - p_pos[0];
                for (std::size_t k = 1; k < n_ratings - 1; ++k) {
                    prob_mat[d_idx][stim][n_ratings + k] = p_pos[k - 1] - p_pos[k];
                }
                prob_mat[d_idx][stim][n_cols - 1] = p_pos[n_ratings - 2];
            }
        }
    }

    return prob_mat;
}

template <typename T>
T ModelNormal<T>::area(
    std::size_t stimulus,
    std::size_t response,
    const T& lower,
    const T& upper,
    std::size_t dim_idx
) const {
    const T d = d_vec[dim_idx];
    const T mu_stim = (stimulus == 0)
        ? (-d * static_cast<T>(0.5))
        : (d * static_cast<T>(0.5));

    if (response == 1) {
        const T p_high_lower = prob_high_conf_normal(mu_stim, c_resp, lower, sigma_meta);
        const T p_high_upper = prob_high_conf_normal(mu_stim, c_resp, upper, sigma_meta);
        return p_high_lower - p_high_upper;
    } else {
        const T p_high_lower = prob_high_conf_normal(-mu_stim, -c_resp, -lower, sigma_meta);
        const T p_high_upper = prob_high_conf_normal(-mu_stim, -c_resp, -upper, sigma_meta);
        return p_high_upper - p_high_lower;
    }
}

template class ModelNormal<double>;
