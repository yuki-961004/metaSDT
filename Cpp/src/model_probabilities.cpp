#include <metaSDT/model_probabilities.hpp>
#include <metaSDT/matrix_prob.hpp>
#include <metaSDT/model_bch.hpp>
#include <metaSDT/model_decay.hpp>
#include <metaSDT/model_lognormal.hpp>
#include <metaSDT/model_normal.hpp>
#include <metaSDT/model_sdt.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/* ========================================================================== *
 *                         Model Probabilities Dispatcher                     *
 * ========================================================================== */

template <typename T>
MatrixProb<T> model_probabilities(
    const std::string& model_id,
    const std::unordered_map<std::string, std::vector<T>>& params
) {
    MatrixProb<T> res;

    if (model_id == "sdt") {
        ModelSDT<T> model(params);
        res = matrix_prob<T>(model.cdf_noise(), model.cdf_signal(), params);
    } else if (model_id == "bch") {
        ModelBCH<T> model(params);
        res = matrix_prob<T>(model.cdf_noise(), model.cdf_signal(), params);
    } else if (model_id == "normal") {
        ModelNormal<T> model(params);
        res.prob_mat = model.compute_probabilities();
    } else if (model_id == "lognormal") {
        ModelLognormal<T> model(params);
        res.prob_mat = model.compute_probabilities();
    } else if (model_id == "decay") {
        ModelDecay<T> model(params);
        res.prob_mat = model.compute_probabilities();
    } else {
        throw std::invalid_argument(
            "Error: Unknown model ID '" + model_id + "'. " +
            "Phase 1 supported models are: 'sdt', 'bch', 'normal', 'lognormal', 'decay'."
        );
    }

    if (res.prob_mat.empty()) {
        throw std::runtime_error("Error: Computed probability matrix is empty.");
    }

    const std::size_t n_diffs = res.prob_mat.size();
    const std::size_t n_cols = res.prob_mat[0][0].size();
    const std::size_t n_conf = n_cols / 2;

    // Apply lapse rate if not already applied by matrix_prob
    if (model_id != "sdt" && model_id != "bch") {
        T lapse = static_cast<T>(0.0);
        auto it_lapse = params.find("rate_lapse");
        if (it_lapse != params.end() && !it_lapse->second.empty()) {
            lapse = it_lapse->second[0];
        }

        if (lapse > static_cast<T>(0.0)) {
            for (std::size_t d = 0; d < n_diffs; ++d) {
                for (std::size_t stim = 0; stim < 2; ++stim) {
                    for (std::size_t j = 0; j < n_cols; ++j) {
                        res.prob_mat[d][stim][j] =
                            (lapse / static_cast<T>(n_cols)) +
                            ((static_cast<T>(1.0) - lapse) * res.prob_mat[d][stim][j]);
                    }
                }
            }
        }

        // Set metadata
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
    }

    // Validation: finite, non-negative, row sum == 1
    using std::abs;
    using std::isfinite;

    for (std::size_t d = 0; d < n_diffs; ++d) {
        for (std::size_t stim = 0; stim < 2; ++stim) {
            T row_sum = static_cast<T>(0.0);
            for (std::size_t j = 0; j < n_cols; ++j) {
                const T p = res.prob_mat[d][stim][j];
                if (!isfinite(p)) {
                    throw std::runtime_error(
                        "Error: Model probability at diff " + std::to_string(d) +
                        ", stim " + std::to_string(stim) + ", col " + std::to_string(j) +
                        " is not finite."
                    );
                }
                if (p < static_cast<T>(0.0)) {
                    // Small floating point underflow clamp
                    res.prob_mat[d][stim][j] = static_cast<T>(0.0);
                }
                row_sum += res.prob_mat[d][stim][j];
            }

            if (abs(row_sum - static_cast<T>(1.0)) > static_cast<T>(1e-4)) {
                throw std::runtime_error(
                    "Error: Model probability row sum for diff " + std::to_string(d) +
                    ", stim " + std::to_string(stim) + " is " + std::to_string(row_sum) +
                    " (expected 1.0)."
                );
            }
        }
    }

    return res;
}

template MatrixProb<double> model_probabilities<double>(
    const std::string&,
    const std::unordered_map<std::string, std::vector<double>>&
);
