#include "../include/modify_outputs.hpp"

#include "../include/criterion_likelihood.hpp"
#include "../include/criterion_posterior.hpp"
#include "../include/matrix_mult.hpp"
#include "../include/matrix_prob.hpp"
#include "../include/model_sdt.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace modify_outputs {

/* ========================================================================== *
 *                         Base Name Ordering Utilities                       *
 * ========================================================================== */

// 构建最终的基础参数名顺序:
// 1) 优先保留用户指定的顺序
// 2) 将 best_params 中未出现过的参数名按字母顺序追加到末尾
std::vector<std::string> ordered_base_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    // 从用户定义的顺序开始, 以保留显式的显示偏好
    std::vector<std::string> out = user_order;

    // 跟踪已放置的参数名, 以避免重复
    std::unordered_map<std::string, bool> seen;
    for (const auto& key : out) {
        seen[key] = true;
    }

    // 收集存在于拟合参数中但未被用户列出的名称.
    std::vector<std::string> remain;
    for (const auto& kv : best_params) {
        if (!seen.count(kv.first)) {
            remain.push_back(kv.first);
        }
    }

    // 对剩余的名称进行排序, 以确保在不同的哈希顺序下输出是确定性的
    std::sort(remain.begin(), remain.end());

    // 将剩余的参数名追加到用户定义的名称之后
    out.insert(out.end(), remain.begin(), remain.end());
    return out;
}

/* ========================================================================== *
 *                        Flat Name Expansion Utilities                       *
 * ========================================================================== */

// 将排序后的基础参数名展开为扁平的列名:
// - 标量参数:   "name"
// - 向量参数:   "name_1", "name_2", ...
std::vector<std::string> ordered_flat_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::size_t>& param_sizes,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    std::vector<std::string> out;

    // 复用基础排序逻辑, 以保持两个 API 的一致性
    std::vector<std::string> bases = ordered_base_names(
        user_order, best_params
    );

    // 根据其有效长度展开每个基础参数名
    for (const auto& base : bases) {
        std::size_t p_len = 0;

        // 优先级 1: 参数模式中声明的长度(如果提供了的话)
        auto its = param_sizes.find(base);
        if (its != param_sizes.end()) {
            p_len = its->second;
        }

        // 优先级 2: 实际拟合的向量长度可能更大; 取两者的最大值
        auto itb = best_params.find(base);
        if (itb != best_params.end()) {
            p_len = std::max(p_len, itb->second.size());
        }

        // 长度为 0 或 1 的参数被视为标量, 以保持紧凑的命名
        if (p_len <= 1) {
            out.push_back(base);
        } else {
            // 多值参数将被展开为基于 1 索引的名称
            for (std::size_t j = 0; j < p_len; ++j) {
                out.push_back(base + "_" + std::to_string(j + 1));
            }
        }
    }
    return out;
}

namespace {

/* ========================================================================== *
 *                              MCMC Helpers                                  *
 * ========================================================================== */

struct MCMCFitStats {
    double logL = 0.0;
    double logPrior = 0.0;
    double logPost = 0.0;
    double aic = 0.0;
    double bic = 0.0;
};

Eigen::VectorXd mcmc_to_eigen(const std::vector<double>& values) {
    Eigen::VectorXd out(static_cast<Eigen::Index>(values.size()));
    for (size_t i = 0; i < values.size(); ++i) {
        out(static_cast<Eigen::Index>(i)) = values[i];
    }
    return out;
}

std::unordered_map<std::string, std::vector<double>> mcmc_params_from_vector(
    const SubjectFitTask& task,
    const std::vector<double>& free_values
) {
    std::unordered_map<std::string, std::vector<double>> out =
        task.params.flat;

    // 按 modify_params 保存的自由参数顺序写回 posterior mean
    task.params.update_map_from_free_vector(out, free_values);

    if (out.count("c_conf") > 0) {
        std::sort(out["c_conf"].begin(), out["c_conf"].end());
    }

    if (out.count("d") > 0 &&
        out.count("sort_d") > 0 &&
        !out["sort_d"].empty() &&
        out["sort_d"][0] != 0.0) {
        std::sort(out["d"].rbegin(), out["d"].rend());
    }

    return out;
}

std::unordered_map<std::string, std::vector<double>> mcmc_sd_from_vector(
    const SubjectFitTask& task,
    const std::vector<double>& sd_values
) {
    std::unordered_map<std::string, std::vector<double>> out;
    size_t cursor = 0;

    // 将扁平 sd 向量拆回参数名, 便于 R/Python 后续表格化
    for (const auto& name : task.params.name_free) {
        const size_t param_len = task.params.structured.free.at(name).size();
        std::vector<double> values(param_len, 0.0);
        for (size_t j = 0; j < param_len; ++j) {
            values[j] = sd_values[cursor++];
        }
        out[name] = values;
    }

    return out;
}

std::unordered_map<std::string, std::vector<std::vector<double>>>
mcmc_samples_by_param(
    const SubjectFitTask& task,
    const std::vector<std::vector<double>>& draws
) {
    std::unordered_map<std::string, std::vector<std::vector<double>>> out;

    for (const auto& name : task.params.name_free) {
        out[name] = {};
        out[name].reserve(draws.size());
    }

    for (const auto& draw : draws) {
        size_t cursor = 0;

        // 每个 draw 依照 name_free 结构拆分, 保留向量参数原始长度
        for (const auto& name : task.params.name_free) {
            const size_t param_len =
                task.params.structured.free.at(name).size();
            std::vector<double> values(param_len, 0.0);
            for (size_t j = 0; j < param_len; ++j) {
                values[j] = draw[cursor++];
            }
            out[name].push_back(values);
        }
    }

    return out;
}

std::vector<double> mcmc_column_mean(
    const std::vector<std::vector<double>>& draws,
    size_t n_dim
) {
    std::vector<double> means(n_dim, 0.0);
    if (draws.empty()) {
        return means;
    }

    for (const auto& draw : draws) {
        for (size_t j = 0; j < n_dim; ++j) {
            means[j] += draw[j];
        }
    }

    for (double& value : means) {
        value /= static_cast<double>(draws.size());
    }

    return means;
}

std::vector<double> mcmc_column_sd(
    const std::vector<std::vector<double>>& draws,
    const std::vector<double>& means
) {
    std::vector<double> sds(means.size(), 0.0);
    if (draws.size() < 2) {
        return sds;
    }

    for (const auto& draw : draws) {
        for (size_t j = 0; j < means.size(); ++j) {
            const double diff = draw[j] - means[j];
            sds[j] += diff * diff;
        }
    }

    for (double& value : sds) {
        value = std::sqrt(value / static_cast<double>(draws.size() - 1));
    }

    return sds;
}

MCMCFitStats evaluate_mcmc_stats(
    const SubjectFitTask& task,
    const std::unordered_map<std::string, std::vector<double>>& params
) {
    const std::vector<double> free_vec =
        task.params.extract_free_vector(params);
    const Eigen::VectorXd free_params = mcmc_to_eigen(free_vec);

    const CriterionPosterior posterior(
        task.freq.freq_mat,
        task.params.name_free,
        task.params.get_free_sizes(),
        task.params.flat,
        task.prior
    );
    const double log_post = posterior.operator()<double>(free_params);

    std::vector<std::vector<double>> cdf_n;
    std::vector<std::vector<double>> cdf_s;

    if (task.model == "sdt") {
        ModelSDT<double> model_obj(params);
        cdf_n = model_obj.cdf_noise();
        cdf_s = model_obj.cdf_signal();
    } else {
        throw std::invalid_argument(
            "Error: Unknown model name '" + task.model + "'."
        );
    }

    const MatrixProb<double> prob = matrix_prob<double>(
        cdf_n,
        cdf_s,
        params
    );
    const auto mult = matrix_mult<double>(
        task.freq.freq_mat,
        prob.prob_mat,
        params
    );
    const auto loss = criterion_likelihood<double>(
        mult,
        task.freq.freq_mat,
        task.params.numb_free,
        free_vec,
        params
    );

    MCMCFitStats out;
    out.logL = loss.logL;
    out.logPost = log_post;
    out.logPrior = out.logPost - out.logL;
    out.aic = loss.aic;
    out.bic = loss.bic;
    return out;
}

} // namespace

/* ========================================================================== *
 *                         MCMC Result Assembly                               *
 * ========================================================================== */

SubjectMCMCResult summarize_mcmc_result(
    const SubjectFitTask& task,
    const std::vector<HMCSamplerResult>& chain_results,
    const StanControl& control,
    int n_evals
) {
    SubjectMCMCResult out;
    out.subid = task.subid;
    out.cond = task.cond;
    out.best_params = task.params.flat;
    out.name_free = task.params.name_free;
    out.size_free = task.params.get_free_sizes();
    out.n_chains = control.chains;
    out.warmup = control.warmup;
    out.thin = control.thin;
    out.leapfrog_steps = control.leapfrog_steps;
    out.n_evals = n_evals;

    std::vector<std::vector<double>> all_draws;
    int status_sum = 0;
    int proposal_sum = 0;
    int accept_sum = 0;
    double step_sum = 0.0;

    for (const HMCSamplerResult& chain : chain_results) {
        all_draws.insert(
            all_draws.end(),
            chain.draws.begin(),
            chain.draws.end()
        );
        out.draw_logPost.insert(
            out.draw_logPost.end(),
            chain.log_prob.begin(),
            chain.log_prob.end()
        );
        status_sum += chain.status > 0 ? 1 : 0;
        proposal_sum += chain.n_proposals;
        accept_sum += chain.n_accept;
        step_sum += chain.final_step_size;
    }

    out.n_draws = static_cast<int>(all_draws.size());
    if (proposal_sum > 0) {
        out.accept_rate =
            static_cast<double>(accept_sum) /
            static_cast<double>(proposal_sum);
    }
    if (!chain_results.empty()) {
        out.step_size =
            step_sum / static_cast<double>(chain_results.size());
    }

    if (all_draws.empty()) {
        out.status = -1;
        out.result_message = "MCMC finished without usable draws.";
        out.stop_reason = "empty_draws";
        return out;
    }

    const size_t n_dim = all_draws[0].size();
    const std::vector<double> means = mcmc_column_mean(all_draws, n_dim);
    const std::vector<double> sds = mcmc_column_sd(all_draws, means);

    out.best_params = mcmc_params_from_vector(task, means);
    out.posterior_sd = mcmc_sd_from_vector(task, sds);
    out.samples = mcmc_samples_by_param(task, all_draws);

    const MCMCFitStats stats = evaluate_mcmc_stats(task, out.best_params);
    out.logL = stats.logL;
    out.logPrior = stats.logPrior;
    out.logPost = stats.logPost;
    out.aic = stats.aic;
    out.bic = stats.bic;

    if (status_sum == static_cast<int>(chain_results.size())) {
        out.status = 1;
        out.result_message = "MCMC sampling finished.";
        out.stop_reason = "complete";
    } else {
        out.status = 0;
        out.result_message = "MCMC sampling finished with chain warnings.";
        out.stop_reason = "partial";
    }

    return out;
}

SubjectMCMCResult failed_mcmc_result(
    const SubjectFitTask& task,
    const StanControl& control,
    const std::string& message,
    const std::string& stop_reason
) {
    SubjectMCMCResult out;
    out.subid = task.subid;
    out.cond = task.cond;
    out.best_params = task.params.flat;
    out.name_free = task.params.name_free;
    out.size_free = task.params.get_free_sizes();
    out.status = -1;
    out.n_chains = control.chains;
    out.warmup = control.warmup;
    out.thin = control.thin;
    out.leapfrog_steps = control.leapfrog_steps;
    out.result_message = message;
    out.stop_reason = stop_reason;
    return out;
}

/* ========================================================================== *
 *                            NLopt Output Slots                              *
 * ========================================================================== */

OutputFitSlot NLopt_fit(
    const std::vector<SubjectFitResult>& results
) {
    OutputFitSlot out;

    // 每个结果对应 fit data frame 的一行, condition 作为外层分组键
    for (const SubjectFitResult& result : results) {
        OutputFitRow row;
        row.subid = result.subid;
        row.cond = result.cond;
        row.logL = result.logL;
        row.logPrior = result.logPrior;
        row.logPost = result.logPost;
        row.aic = result.aic;
        row.bic = result.bic;
        row.status = result.status;
        row.params = result.best_params;

        out.by_condition[result.cond].push_back(row);
    }

    return out;
}

OutputEstimatorSlot NLopt_estimator(
    const std::string& estimator_name,
    const NLoptControl& control
) {
    OutputEstimatorSlot out;
    out.name = estimator_name;
    out.backend = "nlopt";
    out.algorithm = control.algorithm;
    out.global_algorithm = control.algorithm;
    out.local_algorithm = control.local_algorithm;

    // 将控制参数拆成基础类型映射, 方便 R/Python wrapper 统一转换
    out.string_control["algorithm"] = control.algorithm;
    out.string_control["local_algorithm"] = control.local_algorithm;
    out.string_control["progress"] = control.progress;

    out.numeric_control["xtol_rel"] = control.xtol_rel;
    out.numeric_control["maxeval"] = static_cast<double>(control.maxeval);
    out.numeric_control["ftol_rel"] = control.ftol_rel;
    out.numeric_control["ftol_abs"] = control.ftol_abs;
    out.numeric_control["xtol_abs"] = control.xtol_abs;
    out.numeric_control["maxtime"] = control.maxtime;
    out.numeric_control["stopval"] = control.stopval;
    out.numeric_control["population"] =
        static_cast<double>(control.population);
    out.numeric_control["initial_step"] = control.initial_step;
    out.numeric_control["seed"] = static_cast<double>(control.seed);
    out.numeric_control["vector_storage"] =
        static_cast<double>(control.vector_storage);
    out.numeric_control["print_level"] =
        static_cast<double>(control.print_level);
    out.numeric_control["threads"] = static_cast<double>(control.threads);
    out.numeric_control["progress_refresh_ms"] =
        static_cast<double>(control.progress_refresh_ms);
    out.numeric_control["progress_line_interval_sec"] =
        control.progress_line_interval_sec;
    out.numeric_control["progress_line_interval_pct"] =
        control.progress_line_interval_pct;
    out.numeric_control["em_max_iter"] =
        static_cast<double>(control.em_max_iter);
    out.numeric_control["em_tol"] = control.em_tol;
    out.numeric_control["em_patience"] =
        static_cast<double>(control.em_patience);

    out.bool_control["em_init_mle"] = control.em_init_mle;
    out.vector_control["x_weights"] = control.x_weights;

    // nlopt_params 是开放式参数字典, 原样合并进 numeric_control
    for (const auto& kv : control.nlopt_params) {
        out.numeric_control["nlopt." + kv.first] = kv.second;
    }

    return out;
}

OutputDiagnosticsSlot NLopt_diagnostics(
    const std::vector<SubjectFitResult>& results,
    int em_iterations,
    const std::unordered_map<std::string, int>& em_iterations_by_cond,
    const std::unordered_map<std::string, std::string>&
        em_stop_reason_by_cond
) {
    OutputDiagnosticsSlot out;
    out.em_iterations = em_iterations;
    out.em_iterations_by_cond = em_iterations_by_cond;
    out.em_stop_reason_by_cond = em_stop_reason_by_cond;

    // diagnostics 记录搜索过程信息, 不混入 fit 的参数列
    for (const SubjectFitResult& result : results) {
        OutputDiagnosticsRow row;
        row.subid = result.subid;
        row.cond = result.cond;
        row.status = result.status;
        row.n_evals = result.n_evals;
        row.result_code = result.status;
        row.optimum_value = result.optimum_value;
        row.result_message = result.result_message;
        row.stop_reason = result.stop_reason;

        out.by_condition[result.cond].push_back(row);
    }

    return out;
}

/* ========================================================================== *
 *                             Stan Output Slots                              *
 * ========================================================================== */

OutputFitSlot Stan_fit(
    const std::vector<SubjectMCMCResult>& results
) {
    OutputFitSlot out;

    // Stan fit 使用 posterior mean 作为参数列, posterior sd 作为附加列
    for (const SubjectMCMCResult& result : results) {
        OutputFitRow row;
        row.subid = result.subid;
        row.cond = result.cond;
        row.logL = result.logL;
        row.logPrior = result.logPrior;
        row.logPost = result.logPost;
        row.aic = result.aic;
        row.bic = result.bic;
        row.status = result.status;
        row.params = result.best_params;
        row.param_sd = result.posterior_sd;

        out.by_condition[result.cond].push_back(row);
    }

    return out;
}

OutputEstimatorSlot Stan_estimator(
    const std::string& estimator_name,
    const StanControl& control
) {
    OutputEstimatorSlot out;
    const std::string algorithm =
        (control.algorithm == "nuts") ? "NUTS" : "Static HMC";

    out.name = estimator_name;
    out.backend = "stan";
    out.algorithm = algorithm;
    out.global_algorithm = algorithm;
    out.local_algorithm = "";

    // StanControl 也拆成基础类型映射, 让外层 wrapper 只管转格式
    out.string_control["algorithm"] = control.algorithm;
    out.string_control["progress"] = control.progress;

    out.numeric_control["chains"] = static_cast<double>(control.chains);
    out.numeric_control["warmup"] = static_cast<double>(control.warmup);
    out.numeric_control["samples"] = static_cast<double>(control.samples);
    out.numeric_control["thin"] = static_cast<double>(control.thin);
    out.numeric_control["step_size"] = control.step_size;
    out.numeric_control["leapfrog_steps"] =
        static_cast<double>(control.leapfrog_steps);
    out.numeric_control["max_tree_depth"] =
        static_cast<double>(control.max_tree_depth);
    out.numeric_control["target_accept"] = control.target_accept;
    out.numeric_control["min_step_size"] = control.min_step_size;
    out.numeric_control["max_step_size"] = control.max_step_size;
    out.numeric_control["max_delta_energy"] = control.max_delta_energy;
    out.numeric_control["initial_jitter"] = control.initial_jitter;
    out.numeric_control["seed"] = static_cast<double>(control.seed);
    out.numeric_control["print_level"] =
        static_cast<double>(control.print_level);
    out.numeric_control["threads"] = static_cast<double>(control.threads);
    out.numeric_control["progress_refresh_ms"] =
        static_cast<double>(control.progress_refresh_ms);
    out.numeric_control["progress_line_interval_sec"] =
        control.progress_line_interval_sec;
    out.numeric_control["progress_line_interval_pct"] =
        control.progress_line_interval_pct;

    out.bool_control["adapt_step_size"] = control.adapt_step_size;

    return out;
}

OutputDiagnosticsSlot Stan_diagnostics(
    const std::vector<SubjectMCMCResult>& results
) {
    OutputDiagnosticsSlot out;

    // Stan diagnostics 保留采样器状态与 posterior draws, 不进入 fit 表
    for (const SubjectMCMCResult& result : results) {
        OutputDiagnosticsRow row;
        row.subid = result.subid;
        row.cond = result.cond;
        row.status = result.status;
        row.n_evals = result.n_evals;
        row.result_code = result.status;
        row.n_draws = result.n_draws;
        row.n_chains = result.n_chains;
        row.warmup = result.warmup;
        row.thin = result.thin;
        row.leapfrog_steps = result.leapfrog_steps;
        row.accept_rate = result.accept_rate;
        row.step_size = result.step_size;
        row.result_message = result.result_message;
        row.stop_reason = result.stop_reason;
        row.draw_logPost = result.draw_logPost;
        row.samples = result.samples;

        out.by_condition[result.cond].push_back(row);
    }

    return out;
}

/* ========================================================================== *
 *                           Common Output Assembly                           *
 * ========================================================================== */

EstimateOutput combine_output_slots(
    const OutputFitSlot& fit,
    const OutputEstimatorSlot& estimator,
    const OutputDiagnosticsSlot& diagnostics
) {
    EstimateOutput out;
    out.fit = fit;
    out.estimator = estimator;
    out.diagnostics = diagnostics;
    return out;
}

} // namespace modify_outputs
