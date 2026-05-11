#ifndef CRITERION_POSTERIOR_HPP
#define CRITERION_POSTERIOR_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>

#include "criterion_prior.hpp"
#include "model_sdt.hpp"
#include "matrix_prob.hpp"
#include "matrix_mult.hpp"
#include "criterion_likelihood.hpp"

// ==============================================================================
// Unified Log-Posterior Functor
// 完全兼容 Stan Math (Reverse-mode AD), NUTS 采样器，以及 L-BFGS 等优化器
// ==============================================================================
class CriterionPosterior {
private:
    // 客观数据是恒定的且不可导 (强制维持 double)
    std::vector<std::vector<double>> freq_mat_;
    
    // 为了能从一维 Eigen 向量中还原回你层级化的字典结构，我们需要记录映射维度
    std::vector<std::string> param_names_;
    std::vector<int> param_sizes_;
    
    // 不参与梯度优化的固定/常量参数 (如 sim_trials, calc_tol)
    std::unordered_map<std::string, std::vector<double>> static_params_;

    CriterionPrior prior_handler_;

public:
    CriterionPosterior(const std::vector<std::vector<double>>& freq_mat,
                       const std::vector<std::string>& param_names,
                       const std::vector<int>& param_sizes,
                       const std::unordered_map<std::string, 
                                                std::vector<double>>& 
                               static_params,
                       const CriterionPrior& priors)
        : freq_mat_(freq_mat), 
          param_names_(param_names), 
          param_sizes_(param_sizes),
          static_params_(static_params),
          prior_handler_(priors) {}

    // 核心重载符：接受一个未知类型的 Eigen 向量（可以是 double 或 stan::math::var）
    template <typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& free_params) const {
        
        // ==========================================================
        // 1. 评估 Log Prior (计算先验概率密度的对数值)
        // ==========================================================
        // 根据用户事先注入的分布，对当前探索的参数向量 free_params 计算惩罚或引导权重。
        T log_prior = prior_handler_.evaluate<T>(free_params);

        // ==========================================================
        // 2. 参数结构还原 (从一维 Eigen 向量到多维映射字典)
        // ==========================================================
        // Stan、NUTS 或各种 C++ 底层优化器通常只认识一维的纯数值数组/向量。
        // 我们必须将它们重新组装为带有名称和层级结构的字典 (params)，
        // 以便后续模型流水线能够通过名字 (如 "d", "c_conf") 直观、安全地提取。
        std::unordered_map<std::string, std::vector<T>> std_params;
        
        // 2.1 注入固定/常量参数：将 double 隐式升级为 T 
        // (这保证了即使 T 是自动微分变量，常数的梯度也恒为 0)
        for (const auto& kv : static_params_) {
            std::vector<T> vec_t(kv.second.begin(), kv.second.end());
            std_params[kv.first] = vec_t;
        }

        // 2.2 注入自由参数：从一维探索带 free_params 中按预设尺寸逐个抠出变量，拼装成数组
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

        // ==========================================================
        // 3. 前向传播 (Forward Pass)：实例化模型并生成累积分布
        // ==========================================================
        // 把整理好的 params 扔进目标模型 (这里是 SDT)，获取信号和噪声的累积分布边界。
        ModelSDT<T> model(std_params);
        auto cdf_n = model.cdf_noise();
        auto cdf_s = model.cdf_signal();

        // ==========================================================
        // 4. 概率转换与频数对齐 (计算理论概率并与观测频数相乘)
        // ==========================================================
        MatrixProb<T> prob = matrix_prob<T>(cdf_n, cdf_s, std_params);
        auto mult = matrix_mult<T>(freq_mat_, prob.prob_mat, std_params);

        // ==========================================================
        // 5. 似然度与正则化计算 (Likelihood & Regularization)
        // ==========================================================
        // 将理论与观测进行碰撞，得出负对数似然 (NLL)。这其中会顺带附加上可能的 L_n 正则化项。
        int k = free_params.size();
        auto loss = criterion_likelihood<T>(
            mult, freq_mat_, k, 
            free_params_vec, std_params
        );

        // ==========================================================
        // 6. 输出联合未归一化对数后验概率 (Unnormalized Log-Posterior)
        // ==========================================================
        // 根据贝叶斯定理核心思想：Log(Posterior) ∝ Log(Prior) + Log(Likelihood)
        // 注意：如果你需要为 Stan 提供 minimize 目标，应返回负值 (-log_prior - loss.logL)
        // 在此框架下，由于 loss.nll 本身已经是包含了正则化的负对数似然，因此做减法。
        return log_prior - loss.nll;
    }
};

#endif // CRITERION_POSTERIOR_HPP