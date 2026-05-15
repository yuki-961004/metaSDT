#include "../include/estimate_mcmc.hpp"

#include "../include/algorithm_stan.hpp"
#include "../include/task_builder.hpp"
#include "../include/modify_outputs.hpp"
#include "../include/progress_bar.hpp"

#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

/* ========================================================================== *
 *                        Single Subject MCMC Runner                          *
 * ========================================================================== */

SubjectMCMCResult run_subject_mcmc(
    const SubjectFitTask& task,
    const StanControl& control
) {
    if (task.params.numb_free <= 0) {
        return modify_outputs::failed_mcmc_result(
            task,
            control,
            "MCMC requires at least one free parameter.",
            "empty_parameter"
        );
    }

    std::vector<double> initial = task.params.extract_free_vector(
        task.params.flat
    );
    StanAdapter::sanitize_initial_point(
        initial,
        task.params.lower_bounds,
        task.params.upper_bounds
    );

    StanAdapter::Adapter adapter(task);
    const Eigen::VectorXd initial_unconstrained =
        adapter.unconstrain(initial);

    std::vector<HMCSamplerResult> chain_results;
    chain_results.reserve(static_cast<size_t>(control.chains));

    // 每条链顺序运行, 外层被试并行负责利用多核资源.
    for (int chain = 0; chain < control.chains; ++chain) {
        // algorithm 在 modify_control 中已经规范化为 nuts 或 static_hmc.
        if (control.algorithm == "nuts") {
            chain_results.push_back(
                NUTS::run_chain(
                    adapter,
                    initial_unconstrained,
                    control,
                    chain
                )
            );
            continue;
        }

        chain_results.push_back(
            HMC::run_chain(
                adapter,
                initial_unconstrained,
                control,
                chain
            )
        );
    }

    return modify_outputs::summarize_mcmc_result(
        task,
        chain_results,
        control,
        adapter.n_evals()
    );
}

} // namespace

/* ========================================================================== *
 *                              Main Public API                               *
 * ========================================================================== */

std::vector<SubjectMCMCResult> estimate_mcmc(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const StanControl& raw_control,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper,
    const std::unordered_map<std::string, UserPrior>& user_priors
) {
    const StanControl control = modify_control(raw_control, "mcmc");

#ifdef _OPENMP
    // 如果用户指定线程数, 则覆盖 OpenMP 默认线程设置.
    if (control.threads > 0) {
        omp_set_num_threads(control.threads);
    }
#endif

    std::vector<SubjectFitTask> tasks = build_fit_tasks(
        df,
        colnames,
        user_params,
        model_name,
        custom_lower,
        custom_upper,
        true,
        user_priors
    );

    if (tasks.empty()) {
        return {};
    }

    std::vector<SubjectMCMCResult> results(tasks.size());
    const int n_tasks = static_cast<int>(tasks.size());

    if (control.print_level > 0 && n_tasks > 0) {
        ui::ProgressOptions popts;
        popts.mode = control.progress;
        popts.refresh_ms = control.progress_refresh_ms;
        popts.line_interval_sec = control.progress_line_interval_sec;
        popts.line_interval_pct = control.progress_line_interval_pct;
        ui::progress_start(
            static_cast<std::size_t>(n_tasks),
            "MCMC",
            control.progress_refresh_ms,
            popts
        );
    }

    #pragma omp parallel for
    for (int i = 0; i < n_tasks; ++i) {
        const SubjectFitTask& task = tasks[static_cast<size_t>(i)];

        try {
            results[static_cast<size_t>(i)] = run_subject_mcmc(
                task,
                control
            );
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "\n[MCMC Error] Subject " << task.subid
                          << " fitting failed: " << e.what() << "\n";
            }
            results[static_cast<size_t>(i)] =
                modify_outputs::failed_mcmc_result(
                    task,
                    control,
                    e.what(),
                    "exception"
                );
        }

        if (control.print_level > 0) {
            ui::progress_advance(1);
        }
    }

    if (control.print_level > 0 && n_tasks > 0) {
        ui::progress_finish();
    }

    return results;
}
