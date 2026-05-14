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
        T log_prior = prior_handler_.evaluate<T>(free_params);

        std::unordered_map<std::string, std::vector<T>> std_params;

        for (const auto& kv : static_params_) {
            std::vector<T> vec_t(kv.second.begin(), kv.second.end());
            std_params[kv.first] = vec_t;
        }

        int idx = 0;
        std::vector<T> free_params_vec;
        for (size_t i = 0; i < param_names_.size(); ++i) {
            int size = param_sizes_[i];
            std::vector<T> vec_t(size);
            for (int j = 0; j < size; ++j) {
                vec_t[j] = free_params(idx++);
                free_params_vec.push_back(vec_t[j]);
            }
            std_params[param_names_[i]] = vec_t;
        }

        auto it_c_conf = std_params.find("c_conf");
        if (it_c_conf != std_params.end() && !it_c_conf->second.empty()) {
            std::sort(it_c_conf->second.begin(), it_c_conf->second.end());
        }

        auto it_d = std_params.find("d");
        auto it_sort_d = std_params.find("sort_d");
        if (it_d != std_params.end() &&
            it_sort_d != std_params.end() &&
            !it_sort_d->second.empty() &&
            it_sort_d->second[0] != static_cast<T>(0.0)) {
            std::sort(it_d->second.rbegin(), it_d->second.rend());
        }

        ModelSDT<T> model(std_params);
        auto cdf_n = model.cdf_noise();
        auto cdf_s = model.cdf_signal();

        MatrixProb<T> prob = matrix_prob<T>(cdf_n, cdf_s, std_params);
        auto mult = matrix_mult<T>(freq_mat_, prob.prob_mat, std_params);

        int k = static_cast<int>(free_params.size());
        auto loss = criterion_likelihood<T>(
            mult,
            freq_mat_,
            k,
            free_params_vec,
            std_params
        );

        return log_prior - loss.nll;
    }
};

#endif
