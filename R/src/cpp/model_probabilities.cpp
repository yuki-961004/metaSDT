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

    // 所有模型均统一通过 model.area() 接口直接生成概率矩阵
    if (model_id == "sdt") {
        const ModelSDT<T> model(params);
        res = model.area();
    } else if (model_id == "bch") {
        const ModelBCH<T> model(params);
        res = model.area();
    } else if (model_id == "normal") {
        const ModelNormal<T> model(params);
        res = model.area();
    } else if (model_id == "lognormal") {
        const ModelLognormal<T> model(params);
        res = model.area();
    } else if (model_id == "decay") {
        const ModelDecay<T> model(params);
        res = model.area();
    } else {
        throw std::invalid_argument(
            "Error: Unknown model ID '" + model_id + "'. "
            "Phase 1 supported models are: 'sdt', 'bch', 'normal', "
            "'lognormal', 'decay'."
        );
    }

    if (res.prob_mat.empty()) {
        throw std::runtime_error(
            "Error: Computed probability matrix is empty."
        );
    }

    const std::size_t n_diffs = res.prob_mat.size();
    const std::size_t n_cols = res.prob_mat[0][0].size();

    // 严格有效性校验: 有限性、非负性、每行概率和为 1
    for (std::size_t d = 0; d < n_diffs; ++d) {
        for (std::size_t stim = 0; stim < 2; ++stim) {
            T row_sum = static_cast<T>(0.0);
            for (std::size_t j = 0; j < n_cols; ++j) {
                const T p = res.prob_mat[d][stim][j];
                if (!std::isfinite(p)) {
                    throw std::runtime_error(
                        "Error: Model probability at diff " +
                        std::to_string(d) + ", stim " + std::to_string(stim) +
                        ", col " + std::to_string(j) + " is not finite."
                    );
                }
                if (p < static_cast<T>(0.0)) {
                    res.prob_mat[d][stim][j] = static_cast<T>(0.0);
                }
                row_sum += res.prob_mat[d][stim][j];
            }

            if (std::abs(row_sum - static_cast<T>(1.0)) >
                static_cast<T>(1e-3)) {
                throw std::runtime_error(
                    "Error: Model probability row sum for diff " +
                    std::to_string(d) + ", stim " + std::to_string(stim) +
                    " is " + std::to_string(row_sum) + " (expected 1.0)."
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
