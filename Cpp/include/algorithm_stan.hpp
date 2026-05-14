#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "build_objective.hpp"
#include "criterion_posterior.hpp"
#include "modify_control.hpp"

/* ========================================================================== *
 *                             HMC Chain Result                               *
 * ========================================================================== */

struct HMCSamplerResult {
    std::vector<std::vector<double>> draws;
    std::vector<double> log_prob;

    int n_accept = 0;
    int n_proposals = 0;
    int status = -1;

    double accept_rate = 0.0;
    double final_step_size = 0.0;

    std::string result_message;
    std::string stop_reason = "not_started";
};

/* ========================================================================== *
 *                        Stan Posterior Adapter                              *
 * ========================================================================== */

// 将项目内部的 CriterionPosterior 包装成 Stan Math 可求梯度的目标函数
class StanPosteriorAdapter {
public:
    explicit StanPosteriorAdapter(const SubjectFitTask& task);

    // Stan Math finite_diff_gradient_auto 需要 double 向量目标函数
    double operator()(const Eigen::VectorXd& unconstrained) const;

    // 计算 unconstrained 空间中的 log posterior 和梯度
    void value_and_gradient(
        const Eigen::VectorXd& unconstrained,
        double& log_prob,
        Eigen::VectorXd& gradient
    ) const;

    // 从 unconstrained 空间映射到带边界的模型参数空间
    Eigen::VectorXd constrain(
        const Eigen::VectorXd& unconstrained,
        double& log_jacobian
    ) const;

    // 将 constrained 参数向量映射回 unconstrained 空间
    Eigen::VectorXd unconstrain(
        const std::vector<double>& constrained
    ) const;

    // 用于采样输出, 直接返回 std::vector 格式的 constrained 参数
    std::vector<double> constrain_to_vector(
        const Eigen::VectorXd& unconstrained
    ) const;

    int n_evals() const;

private:
    CriterionPosterior posterior_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    mutable int n_evals_ = 0;
};

/* ========================================================================== *
 *                       Initial Value Bound Handling                         *
 * ========================================================================== */

// 将初始值压回合法边界内部, 避免 unconstrain 时落在边界正上方
void stan_sanitize_initial_point(
    std::vector<double>& initial,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double epsilon = 1e-6
);

/* ========================================================================== *
 *                            HMC Sampler Entry                               *
 * ========================================================================== */

// 使用 StanPosteriorAdapter 提供的 log posterior 和梯度运行单条 HMC 链
HMCSamplerResult run_hmc_chain(
    const StanPosteriorAdapter& adapter,
    const Eigen::VectorXd& initial_unconstrained,
    const StanControl& control,
    int chain_id
);

// 使用 No-U-Turn 动态树深度规则运行单条 NUTS 链
HMCSamplerResult run_nuts_chain(
    const StanPosteriorAdapter& adapter,
    const Eigen::VectorXd& initial_unconstrained,
    const StanControl& control,
    int chain_id
);
