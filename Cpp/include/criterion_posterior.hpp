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

// 专为 MCMC 采样和 EM-MAP 算法设计的后验概率计算引擎
// 给定数据和参数, 能够自动结合先验与似然计算总对数后验
// 当前在 estimate_map 模块中被调用以评估优化路径
class CriterionPosterior {
private:
    std::vector<std::vector<std::vector<double>>> freq_mat_;
    std::vector<std::string> param_names_;
    std::vector<int> param_sizes_;
    std::unordered_map<std::string, std::vector<double>> static_params_;
    CriterionPrior prior_handler_;

public:
/* ========================================================================== *
 *                      Initialization & Data Binding                         *
 * ========================================================================== */

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

/* ========================================================================== *
 *                  Core Functor: Log-Posterior Evaluation                    *
 * ========================================================================== */

    template <typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& free_params) const {
        // 计算当前自由参数的对数先验概率 (Log-Prior)
        T log_prior = prior_handler_.evaluate<T>(free_params);

        std::unordered_map<std::string, std::vector<T>> std_params;

        // 载入不参与求导与采样的静态参数 (fixed / constant)
        for (const auto& kv : static_params_) {
            std::vector<T> vec_t(kv.second.begin(), kv.second.end());
            std_params[kv.first] = vec_t;
        }

        // 将传入的扁平化一维自由参数向量映射回结构化参数字典
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

        // 强制确立参数的单调性约束, 保证置信阈值有序
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

        // 构建底层 SDT 模型并推导累积分布矩阵
        ModelSDT<T> model(std_params);
        auto cdf_n = model.cdf_noise();
        auto cdf_s = model.cdf_signal();

        // 生成理论概率矩阵并结合观测频数构造似然输入
        MatrixProb<T> prob = matrix_prob<T>(cdf_n, cdf_s, std_params);
        auto mult = matrix_mult<T>(freq_mat_, prob.prob_mat, std_params);

        int k = static_cast<int>(free_params.size());
        // 评估负对数似然 (NLL) 等统计评价指标
        auto loss = criterion_likelihood<T>(
            mult,
            freq_mat_,
            k,
            free_params_vec,
            std_params
        );

        // 返回最终的目标计算值: LogPosterior = LogPrior - NLL
        return log_prior - loss.nll;
    }
};

#endif
