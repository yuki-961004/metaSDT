#ifndef CRITERION_POSTERIOR_HPP
#define CRITERION_POSTERIOR_HPP

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "criterion_likelihood.hpp"
#include "criterion_prior.hpp"
#include "matrix_mult.hpp"
#include "matrix_prob.hpp"
#include "model_sdt.hpp"

/* ========================================================================== *
 *                       Criterion Posterior Engine                           *
 * ========================================================================== */

class CriterionPosterior {
private:
    std::vector<std::vector<std::vector<double>>> freq_mat_;
    std::vector<std::string> param_names_;
    std::vector<int> param_sizes_;
    std::unordered_map<std::string, std::vector<double>> static_params_;
    CriterionPrior prior_handler_;

public:
    CriterionPosterior(
        const std::vector<std::vector<std::vector<double>>>& freq_mat,
        const std::vector<std::string>& param_names,
        const std::vector<int>& param_sizes,
        const std::unordered_map<std::string, std::vector<double>>&
            static_params,
        const CriterionPrior& priors
    )
        : freq_mat_(freq_mat),
          param_names_(param_names),
          param_sizes_(param_sizes),
          static_params_(static_params),
          prior_handler_(priors) {}

    template <typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& free_params) const {
        // 先计算当前自由参数向量对应的对数先验.
        T log_prior = prior_handler_.evaluate<T>(free_params);

        std::unordered_map<std::string, std::vector<T>> std_params;

        // 固定参数和常数参数直接复制到运行时参数表里.
        for (const auto& kv : static_params_) {
            std::vector<T> values(kv.second.begin(), kv.second.end());
            std_params[kv.first] = values;
        }

        // 按构建任务时记录的顺序, 把扁平自由参数还原成命名向量.
        int flat_index = 0;
        std::vector<T> free_params_vec;
        for (std::size_t index = 0; index < param_names_.size(); ++index) {
            const int param_size = param_sizes_[index];
            std::vector<T> values(static_cast<std::size_t>(param_size));

            for (int inner = 0; inner < param_size; ++inner) {
                values[static_cast<std::size_t>(inner)] = free_params(flat_index);
                free_params_vec.push_back(free_params(flat_index));
                ++flat_index;
            }

            std_params[param_names_[index]] = values;
        }

        // 置信标准必须保持单调递增, 避免模型概率矩阵顺序错乱.
        auto it_c_conf = std_params.find("c_conf");
        if (it_c_conf != std_params.end() && !it_c_conf->second.empty()) {
            std::sort(it_c_conf->second.begin(), it_c_conf->second.end());
        }

        // 当 sort_d 打开时, 难度敏感度按降序排列以保持模型约束.
        auto it_d = std_params.find("d");
        auto it_sort_d = std_params.find("sort_d");
        if (it_d != std_params.end() &&
            it_sort_d != std_params.end() &&
            !it_sort_d->second.empty() &&
            it_sort_d->second[0] != static_cast<T>(0.0)) {
            std::sort(it_d->second.rbegin(), it_d->second.rend());
        }

        // 由当前参数生成模型概率, 再与频数矩阵组合成似然项.
        ModelSDT<T> model(std_params);
        auto cdf_n = model.cdf_noise();
        auto cdf_s = model.cdf_signal();
        MatrixProb<T> prob = matrix_prob<T>(cdf_n, cdf_s, std_params);
        auto mult = matrix_mult<T>(freq_mat_, prob.prob_mat, std_params);

        const int n_free = static_cast<int>(free_params.size());
        auto loss = criterion_likelihood<T>(
            mult,
            freq_mat_,
            n_free,
            free_params_vec,
            std_params
        );

        // 后验目标为 log prior - negative log likelihood.
        return log_prior - loss.nll;
    }
};

#endif // CRITERION_POSTERIOR_HPP
