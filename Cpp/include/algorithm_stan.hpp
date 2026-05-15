#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "criterion_posterior.hpp"
#include "modify_control.hpp"
#include "task_builder.hpp"

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
 *                           Stan Math Adapter                                *
 * ========================================================================== */

namespace StanAdapter {

// Adapter between project criteria and Stan Math gradient utilities.
// 它像转换插头, 一边接项目自己的后验准则, 一边接梯度计算工具.
class Adapter {
public:
    explicit Adapter(const SubjectFitTask& task);

    // Return log posterior in the unconstrained space.
    // 名字和 NLoptAdapter::criterion 对齐, 但这里返回的是采样密度.
    double criterion(const Eigen::VectorXd& unconstrained) const;

    // Calculate criterion value and finite-difference gradient.
    // 采样器每迈一步都要问这里, 当前山坡的高度和坡度是多少.
    void gradient(
        const Eigen::VectorXd& unconstrained,
        double& log_prob,
        Eigen::VectorXd& gradient
    ) const;

    // Map unconstrained sampler coordinates to bounded model parameters.
    Eigen::VectorXd constrain(
        const Eigen::VectorXd& unconstrained,
        double& log_jacobian
    ) const;

    // Map bounded model parameters back to unconstrained sampler coordinates.
    Eigen::VectorXd unconstrain(
        const std::vector<double>& constrained
    ) const;

    // Convert an unconstrained draw into a plain vector for output tables.
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

// Keep initial values strictly inside legal model bounds.
void sanitize_initial_point(
    std::vector<double>& initial,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double epsilon = 1e-6
);

} // namespace StanAdapter

/* ========================================================================== *
 *                                HMC Runner                                  *
 * ========================================================================== */

namespace HMC {

// Run one static HMC chain using StanAdapter::Adapter as the log density.
HMCSamplerResult run_chain(
    const StanAdapter::Adapter& adapter,
    const Eigen::VectorXd& initial_unconstrained,
    const StanControl& control,
    int chain_id
);

} // namespace HMC

/* ========================================================================== *
 *                                NUTS Runner                                 *
 * ========================================================================== */

namespace NUTS {

// Run one NUTS chain using StanAdapter::Adapter as the log density.
HMCSamplerResult run_chain(
    const StanAdapter::Adapter& adapter,
    const Eigen::VectorXd& initial_unconstrained,
    const StanControl& control,
    int chain_id
);

} // namespace NUTS
