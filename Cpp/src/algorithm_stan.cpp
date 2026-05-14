#include "../include/algorithm_stan.hpp"

#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/functor/finite_diff_gradient_auto.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

/* ========================================================================== *
 *                              Numeric Helpers                               *
 * ========================================================================== */

double stan_logit(double probability) {
    return std::log(probability) - std::log1p(-probability);
}

double stan_inv_logit(double value) {
    if (value >= 0.0) {
        const double exp_neg = std::exp(-value);
        return 1.0 / (1.0 + exp_neg);
    }

    const double exp_pos = std::exp(value);
    return exp_pos / (1.0 + exp_pos);
}

double stan_log_inv_logit(double value) {
    if (value >= 0.0) {
        return -std::log1p(std::exp(-value));
    }
    return value - std::log1p(std::exp(value));
}

double stan_log1m_inv_logit(double value) {
    if (value >= 0.0) {
        return -value - std::log1p(std::exp(-value));
    }
    return -std::log1p(std::exp(value));
}

double stan_clamp_probability(double value) {
    const double eps = 1e-12;
    return std::max(eps, std::min(value, 1.0 - eps));
}

bool stan_is_finite(double value) {
    return std::isfinite(value);
}

bool hmc_is_finite_vector(const Eigen::VectorXd& values) {
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values(i))) {
            return false;
        }
    }
    return true;
}

double hmc_safe_accept_probability(double log_accept_ratio) {
    if (!std::isfinite(log_accept_ratio)) {
        return 0.0;
    }
    if (log_accept_ratio >= 0.0) {
        return 1.0;
    }
    return std::exp(log_accept_ratio);
}

Eigen::VectorXd hmc_draw_momentum(
    Eigen::Index n_dim,
    std::mt19937_64& rng
) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd momentum(n_dim);

    // 每个维度独立抽取标准正态动量, 对应单位质量矩阵
    for (Eigen::Index i = 0; i < n_dim; ++i) {
        momentum(i) = normal(rng);
    }

    return momentum;
}

void hmc_jitter_initial(
    Eigen::VectorXd& initial,
    double jitter,
    std::mt19937_64& rng
) {
    if (jitter <= 0.0) {
        return;
    }

    std::normal_distribution<double> normal(0.0, jitter);

    // 多链从轻微不同的起点出发, 降低完全同轨迹的风险
    for (Eigen::Index i = 0; i < initial.size(); ++i) {
        initial(i) += normal(rng);
    }
}

} // namespace

/* ========================================================================== *
 *                        Stan Posterior Adapter                              *
 * ========================================================================== */

StanPosteriorAdapter::StanPosteriorAdapter(const SubjectFitTask& task)
    : posterior_(
          task.freq.freq_mat,
          task.params.name_free,
          task.params.get_free_sizes(),
          task.params.flat,
          task.prior
      ),
      lower_bounds_(task.params.lower_bounds),
      upper_bounds_(task.params.upper_bounds) {}

double StanPosteriorAdapter::operator()(
    const Eigen::VectorXd& unconstrained
) const {
    ++n_evals_;

    double log_jacobian = 0.0;
    const Eigen::VectorXd constrained = constrain(
        unconstrained,
        log_jacobian
    );

    // 如果参数转换已经失败, 则返回负无穷使 HMC 自动拒绝
    if (!stan_is_finite(log_jacobian)) {
        return -std::numeric_limits<double>::infinity();
    }

    const double log_posterior = posterior_.operator()<double>(constrained);

    // CriterionPosterior 的值必须有限, 否则该 proposal 无效
    if (!stan_is_finite(log_posterior)) {
        return -std::numeric_limits<double>::infinity();
    }

    return log_posterior + log_jacobian;
}

void StanPosteriorAdapter::value_and_gradient(
    const Eigen::VectorXd& unconstrained,
    double& log_prob,
    Eigen::VectorXd& gradient
) const {
    gradient.resize(unconstrained.size());

    // Stan Math 在这里负责高阶有限差分梯度, 目标函数仍复用项目后验
    Eigen::VectorXd work = unconstrained;
    stan::math::finite_diff_gradient_auto(
        *this,
        work,
        log_prob,
        gradient
    );
}

Eigen::VectorXd StanPosteriorAdapter::constrain(
    const Eigen::VectorXd& unconstrained,
    double& log_jacobian
) const {
    const Eigen::Index n_dim = unconstrained.size();
    Eigen::VectorXd constrained(n_dim);
    log_jacobian = 0.0;

    for (Eigen::Index i = 0; i < n_dim; ++i) {
        const double z = unconstrained(i);
        const double lower = lower_bounds_[static_cast<size_t>(i)];
        const double upper = upper_bounds_[static_cast<size_t>(i)];
        const bool has_lower = stan_is_finite(lower);
        const bool has_upper = stan_is_finite(upper);

        if (has_lower && has_upper) {
            const double width = upper - lower;

            // 上下界必须形成有效区间, 否则该目标函数无定义
            if (width <= 0.0) {
                log_jacobian = -std::numeric_limits<double>::infinity();
                constrained(i) = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            const double probability = stan_inv_logit(z);
            constrained(i) = lower + width * probability;
            log_jacobian += std::log(width);
            log_jacobian += stan_log_inv_logit(z);
            log_jacobian += stan_log1m_inv_logit(z);
        } else if (has_lower) {
            constrained(i) = lower + std::exp(z);
            log_jacobian += z;
        } else if (has_upper) {
            constrained(i) = upper - std::exp(z);
            log_jacobian += z;
        } else {
            constrained(i) = z;
        }
    }

    return constrained;
}

Eigen::VectorXd StanPosteriorAdapter::unconstrain(
    const std::vector<double>& constrained
) const {
    Eigen::VectorXd unconstrained(
        static_cast<Eigen::Index>(constrained.size())
    );

    for (size_t i = 0; i < constrained.size(); ++i) {
        const double value = constrained[i];
        const double lower = lower_bounds_[i];
        const double upper = upper_bounds_[i];
        const bool has_lower = stan_is_finite(lower);
        const bool has_upper = stan_is_finite(upper);

        if (has_lower && has_upper) {
            const double width = upper - lower;
            if (width <= 0.0) {
                throw std::invalid_argument(
                    "Stan transform received invalid parameter bounds."
                );
            }

            const double probability = stan_clamp_probability(
                (value - lower) / width
            );
            unconstrained(static_cast<Eigen::Index>(i)) =
                stan_logit(probability);
        } else if (has_lower) {
            unconstrained(static_cast<Eigen::Index>(i)) =
                std::log(std::max(value - lower, 1e-12));
        } else if (has_upper) {
            unconstrained(static_cast<Eigen::Index>(i)) =
                std::log(std::max(upper - value, 1e-12));
        } else {
            unconstrained(static_cast<Eigen::Index>(i)) = value;
        }
    }

    return unconstrained;
}

std::vector<double> StanPosteriorAdapter::constrain_to_vector(
    const Eigen::VectorXd& unconstrained
) const {
    double log_jacobian = 0.0;
    const Eigen::VectorXd constrained = constrain(
        unconstrained,
        log_jacobian
    );

    std::vector<double> out(static_cast<size_t>(constrained.size()));
    for (Eigen::Index i = 0; i < constrained.size(); ++i) {
        out[static_cast<size_t>(i)] = constrained(i);
    }

    return out;
}

int StanPosteriorAdapter::n_evals() const {
    return n_evals_;
}

/* ========================================================================== *
 *                       Initial Value Bound Handling                         *
 * ========================================================================== */

void stan_sanitize_initial_point(
    std::vector<double>& initial,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double epsilon
) {
    for (size_t i = 0; i < initial.size(); ++i) {
        const bool has_lower = stan_is_finite(lower_bounds[i]);
        const bool has_upper = stan_is_finite(upper_bounds[i]);

        if (has_lower && initial[i] <= lower_bounds[i]) {
            // 有下界时, 初始值必须推入可行域内部
            initial[i] = lower_bounds[i] + epsilon;
        }

        if (has_upper && initial[i] >= upper_bounds[i]) {
            // 有上界时, 初始值必须推入可行域内部
            initial[i] = upper_bounds[i] - epsilon;
        }

        if (has_lower && has_upper) {
            const double midpoint =
                0.5 * (lower_bounds[i] + upper_bounds[i]);

            // 如果边界过窄导致修正仍不合法, 则回退到区间中点
            if (initial[i] <= lower_bounds[i] ||
                initial[i] >= upper_bounds[i]) {
                initial[i] = midpoint;
            }
        }
    }
}

namespace {

/* ========================================================================== *
 *                             NUTS Tree Logic                                *
 * ========================================================================== */

struct NUTSState {
    Eigen::VectorXd position;
    Eigen::VectorXd momentum;
    Eigen::VectorXd gradient;
    double log_prob = -std::numeric_limits<double>::infinity();
    bool valid = false;
};

struct NUTSTreeResult {
    NUTSState left;
    NUTSState right;
    NUTSState candidate;

    int n_valid = 0;
    int accept_count = 0;

    double accept_sum = 0.0;

    bool valid_candidate = false;
    bool continue_tree = false;
};

bool nuts_no_u_turn(
    const NUTSState& left,
    const NUTSState& right
) {
    const Eigen::VectorXd delta = right.position - left.position;
    return delta.dot(left.momentum) >= 0.0 &&
        delta.dot(right.momentum) >= 0.0;
}

NUTSState nuts_leapfrog(
    const StanPosteriorAdapter& adapter,
    const NUTSState& start,
    double step_size,
    int direction
) {
    NUTSState next;
    const double signed_step =
        static_cast<double>(direction) * step_size;

    // 半步更新动量, 再整步更新位置, 最后用新梯度补齐半步动量
    Eigen::VectorXd momentum =
        start.momentum + 0.5 * signed_step * start.gradient;
    next.position = start.position + signed_step * momentum;

    adapter.value_and_gradient(
        next.position,
        next.log_prob,
        next.gradient
    );

    if (!std::isfinite(next.log_prob) ||
        !hmc_is_finite_vector(next.gradient)) {
        next.valid = false;
        return next;
    }

    next.momentum = momentum + 0.5 * signed_step * next.gradient;
    next.valid = hmc_is_finite_vector(next.position) &&
        hmc_is_finite_vector(next.momentum);
    return next;
}

NUTSTreeResult nuts_build_tree(
    const StanPosteriorAdapter& adapter,
    const NUTSState& state,
    int direction,
    int depth,
    double log_slice,
    double initial_joint,
    double step_size,
    double max_delta_energy,
    std::mt19937_64& rng
) {
    if (depth == 0) {
        NUTSTreeResult out;
        const NUTSState next = nuts_leapfrog(
            adapter,
            state,
            step_size,
            direction
        );
        out.left = next;
        out.right = next;
        out.candidate = next;
        out.accept_count = 1;

        if (!next.valid) {
            out.continue_tree = false;
            return out;
        }

        const double joint =
            next.log_prob - 0.5 * next.momentum.squaredNorm();
        out.n_valid = (log_slice <= joint) ? 1 : 0;
        out.valid_candidate = out.n_valid > 0;
        out.continue_tree = (log_slice - max_delta_energy) < joint;
        out.accept_sum = hmc_safe_accept_probability(
            joint - initial_joint
        );
        return out;
    }

    NUTSTreeResult first = nuts_build_tree(
        adapter,
        state,
        direction,
        depth - 1,
        log_slice,
        initial_joint,
        step_size,
        max_delta_energy,
        rng
    );

    if (!first.continue_tree) {
        return first;
    }

    const NUTSState& edge = (direction == -1) ? first.left : first.right;
    NUTSTreeResult second = nuts_build_tree(
        adapter,
        edge,
        direction,
        depth - 1,
        log_slice,
        initial_joint,
        step_size,
        max_delta_energy,
        rng
    );

    NUTSTreeResult out;
    if (direction == -1) {
        out.left = second.left;
        out.right = first.right;
    } else {
        out.left = first.left;
        out.right = second.right;
    }

    out.candidate = first.candidate;
    out.valid_candidate = first.valid_candidate;

    const int combined_valid = first.n_valid + second.n_valid;
    if (second.valid_candidate && combined_valid > 0) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double choose_second =
            static_cast<double>(second.n_valid) /
            static_cast<double>(combined_valid);

        // 用子树有效点数量作为权重, 保持从切片集合中近似均匀选择
        if (!out.valid_candidate || uniform(rng) < choose_second) {
            out.candidate = second.candidate;
            out.valid_candidate = true;
        }
    }

    out.n_valid = combined_valid;
    out.accept_count = first.accept_count + second.accept_count;
    out.accept_sum = first.accept_sum + second.accept_sum;
    out.continue_tree = first.continue_tree &&
        second.continue_tree &&
        nuts_no_u_turn(out.left, out.right);
    return out;
}

} // namespace

/* ========================================================================== *
 *                            HMC Sampler Entry                               *
 * ========================================================================== */

HMCSamplerResult run_hmc_chain(
    const StanPosteriorAdapter& adapter,
    const Eigen::VectorXd& initial_unconstrained,
    const StanControl& control,
    int chain_id
) {
    HMCSamplerResult out;
    out.final_step_size = control.step_size;

    const Eigen::Index n_dim = initial_unconstrained.size();
    if (n_dim <= 0) {
        out.status = -1;
        out.result_message = "MCMC requires at least one free parameter.";
        out.stop_reason = "empty_parameter";
        return out;
    }

    const unsigned long long chain_seed =
        static_cast<unsigned long long>(control.seed) +
        static_cast<unsigned long long>(chain_id) * 104729ULL;
    std::mt19937_64 rng(chain_seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    Eigen::VectorXd current = initial_unconstrained;
    hmc_jitter_initial(current, control.initial_jitter, rng);

    double current_log_prob = 0.0;
    Eigen::VectorXd current_gradient(n_dim);
    adapter.value_and_gradient(
        current,
        current_log_prob,
        current_gradient
    );

    if (!std::isfinite(current_log_prob) ||
        !hmc_is_finite_vector(current_gradient)) {
        out.status = -1;
        out.result_message = "Initial MCMC state has invalid log density.";
        out.stop_reason = "invalid_initial_state";
        return out;
    }

    double step_size = control.step_size;
    const int total_iterations =
        control.warmup + control.samples * control.thin;
    out.draws.reserve(static_cast<size_t>(control.samples));
    out.log_prob.reserve(static_cast<size_t>(control.samples));

    for (int iter = 0; iter < total_iterations; ++iter) {
        const Eigen::VectorXd initial_momentum =
            hmc_draw_momentum(n_dim, rng);
        Eigen::VectorXd proposed = current;
        Eigen::VectorXd proposed_momentum = initial_momentum;
        Eigen::VectorXd proposed_gradient = current_gradient;
        double proposed_log_prob = current_log_prob;
        bool valid_proposal = true;

        // leapfrog 使用 log posterior 梯度, 所以动量沿上升方向更新
        proposed_momentum += 0.5 * step_size * proposed_gradient;

        for (int leap = 0; leap < control.leapfrog_steps; ++leap) {
            proposed += step_size * proposed_momentum;

            adapter.value_and_gradient(
                proposed,
                proposed_log_prob,
                proposed_gradient
            );

            if (!std::isfinite(proposed_log_prob) ||
                !hmc_is_finite_vector(proposed_gradient)) {
                valid_proposal = false;
                break;
            }

            // 最后一步只做半步动量更新, 保持 leapfrog 对称性
            if (leap + 1 < control.leapfrog_steps) {
                proposed_momentum += step_size * proposed_gradient;
            }
        }

        double accept_probability = 0.0;
        if (valid_proposal) {
            proposed_momentum += 0.5 * step_size * proposed_gradient;
            proposed_momentum = -proposed_momentum;

            const double current_energy =
                -current_log_prob +
                0.5 * initial_momentum.squaredNorm();
            const double proposed_energy =
                -proposed_log_prob +
                0.5 * proposed_momentum.squaredNorm();
            const double log_accept_ratio =
                current_energy - proposed_energy;
            accept_probability =
                hmc_safe_accept_probability(log_accept_ratio);

            if (std::log(uniform(rng)) < log_accept_ratio) {
                current = proposed;
                current_log_prob = proposed_log_prob;
                current_gradient = proposed_gradient;
                out.n_accept += 1;
            }
        }

        out.n_proposals += 1;

        if (iter < control.warmup && control.adapt_step_size) {
            const double adapt_rate =
                1.0 / std::sqrt(static_cast<double>(iter + 1));
            const double log_step =
                std::log(step_size) +
                adapt_rate * (accept_probability - control.target_accept);
            step_size = std::exp(log_step);
            step_size = std::max(
                control.min_step_size,
                std::min(step_size, control.max_step_size)
            );
        }

        if (iter >= control.warmup) {
            const int sampling_iter = iter - control.warmup;

            // thin 控制保留间隔, 其余迭代只用于推进 Markov chain
            if (sampling_iter % control.thin == 0) {
                out.draws.push_back(adapter.constrain_to_vector(current));
                out.log_prob.push_back(current_log_prob);
            }
        }
    }

    out.final_step_size = step_size;
    if (out.n_proposals > 0) {
        out.accept_rate =
            static_cast<double>(out.n_accept) /
            static_cast<double>(out.n_proposals);
    }

    if (static_cast<int>(out.draws.size()) == control.samples) {
        out.status = 1;
        out.result_message = "HMC sampling finished.";
        out.stop_reason = "complete";
    } else {
        out.status = -1;
        out.result_message = "HMC produced fewer draws than needed.";
        out.stop_reason = "insufficient_draws";
    }

    return out;
}

/* ========================================================================== *
 *                            NUTS Sampler Entry                              *
 * ========================================================================== */

HMCSamplerResult run_nuts_chain(
    const StanPosteriorAdapter& adapter,
    const Eigen::VectorXd& initial_unconstrained,
    const StanControl& control,
    int chain_id
) {
    HMCSamplerResult out;
    out.final_step_size = control.step_size;

    const Eigen::Index n_dim = initial_unconstrained.size();
    if (n_dim <= 0) {
        out.status = -1;
        out.result_message = "NUTS requires at least one free parameter.";
        out.stop_reason = "empty_parameter";
        return out;
    }

    const unsigned long long chain_seed =
        static_cast<unsigned long long>(control.seed) +
        static_cast<unsigned long long>(chain_id) * 104729ULL;
    std::mt19937_64 rng(chain_seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::exponential_distribution<double> exponential(1.0);

    NUTSState current;
    current.position = initial_unconstrained;
    hmc_jitter_initial(current.position, control.initial_jitter, rng);
    adapter.value_and_gradient(
        current.position,
        current.log_prob,
        current.gradient
    );
    current.valid = std::isfinite(current.log_prob) &&
        hmc_is_finite_vector(current.gradient);

    if (!current.valid) {
        out.status = -1;
        out.result_message = "Initial NUTS state has invalid log density.";
        out.stop_reason = "invalid_initial_state";
        return out;
    }

    double step_size = control.step_size;
    double accept_rate_sum = 0.0;
    const int total_iterations =
        control.warmup + control.samples * control.thin;
    out.draws.reserve(static_cast<size_t>(control.samples));
    out.log_prob.reserve(static_cast<size_t>(control.samples));

    for (int iter = 0; iter < total_iterations; ++iter) {
        current.momentum = hmc_draw_momentum(n_dim, rng);
        const double initial_joint =
            current.log_prob - 0.5 * current.momentum.squaredNorm();
        const double log_slice = initial_joint - exponential(rng);

        NUTSState left = current;
        NUTSState right = current;
        NUTSState proposal = current;

        int n_valid = 1;
        int depth = 0;
        bool keep_sampling = true;
        double accept_sum = 0.0;
        int accept_count = 0;

        while (keep_sampling && depth < control.max_tree_depth) {
            const int direction = (uniform(rng) < 0.5) ? -1 : 1;
            NUTSTreeResult tree;

            if (direction == -1) {
                tree = nuts_build_tree(
                    adapter,
                    left,
                    direction,
                    depth,
                    log_slice,
                    initial_joint,
                    step_size,
                    control.max_delta_energy,
                    rng
                );
                left = tree.left;
            } else {
                tree = nuts_build_tree(
                    adapter,
                    right,
                    direction,
                    depth,
                    log_slice,
                    initial_joint,
                    step_size,
                    control.max_delta_energy,
                    rng
                );
                right = tree.right;
            }

            if (tree.valid_candidate && tree.n_valid > 0) {
                const int combined_valid = n_valid + tree.n_valid;
                const double choose_tree =
                    static_cast<double>(tree.n_valid) /
                    static_cast<double>(combined_valid);

                // 按有效候选数量合并旧树与新树, 避免偏向较早构造的节点
                if (uniform(rng) < choose_tree) {
                    proposal = tree.candidate;
                }
                n_valid = combined_valid;
            }

            accept_sum += tree.accept_sum;
            accept_count += tree.accept_count;
            keep_sampling = tree.continue_tree &&
                nuts_no_u_turn(left, right);
            ++depth;
        }

        if (proposal.valid) {
            current.position = proposal.position;
            current.log_prob = proposal.log_prob;
            current.gradient = proposal.gradient;
        }

        const double accept_probability = (accept_count > 0)
            ? accept_sum / static_cast<double>(accept_count)
            : 0.0;
        accept_rate_sum += accept_probability;
        out.n_proposals += 1;

        if (iter < control.warmup && control.adapt_step_size) {
            const double adapt_rate =
                1.0 / std::sqrt(static_cast<double>(iter + 1));
            const double log_step =
                std::log(step_size) +
                adapt_rate * (accept_probability - control.target_accept);
            step_size = std::exp(log_step);
            step_size = std::max(
                control.min_step_size,
                std::min(step_size, control.max_step_size)
            );
        }

        if (iter >= control.warmup) {
            const int sampling_iter = iter - control.warmup;
            if (sampling_iter % control.thin == 0) {
                out.draws.push_back(
                    adapter.constrain_to_vector(current.position)
                );
                out.log_prob.push_back(current.log_prob);
            }
        }
    }

    out.final_step_size = step_size;
    if (out.n_proposals > 0) {
        out.accept_rate =
            accept_rate_sum / static_cast<double>(out.n_proposals);
    }

    if (static_cast<int>(out.draws.size()) == control.samples) {
        out.status = 1;
        out.result_message = "NUTS sampling finished.";
        out.stop_reason = "complete";
    } else {
        out.status = -1;
        out.result_message = "NUTS produced fewer draws than needed.";
        out.stop_reason = "insufficient_draws";
    }

    return out;
}
