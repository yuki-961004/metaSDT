#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "algorithm_stan.hpp"
#include "task_builder.hpp"
#include "modify_control.hpp"

/* ========================================================================== *
 *                          Subject MCMC Result                               *
 * ========================================================================== */

struct SubjectMCMCResult {
    double subid = 0.0;
    std::string cond;

    double logL = 0.0;
    double logPrior = 0.0;
    double logPost = 0.0;
    double aic = 0.0;
    double bic = 0.0;

    std::unordered_map<std::string, std::vector<double>> best_params;
    std::unordered_map<std::string, std::vector<double>> posterior_sd;
    std::unordered_map<std::string, std::vector<std::vector<double>>> samples;

    std::vector<std::string> name_free;
    std::vector<int> size_free;
    std::vector<double> draw_logPost;

    int status = -1;
    int n_evals = 0;
    int n_draws = 0;
    int n_chains = 0;
    int warmup = 0;
    int thin = 1;
    int leapfrog_steps = 0;

    double accept_rate = 0.0;
    double step_size = 0.0;

    std::string result_message;
    std::string stop_reason = "not_started";
};

/* ========================================================================== *
 *                         Common Output Slot Types                           *
 * ========================================================================== */

struct OutputFitRow {
    double subid = 0.0;
    std::string cond;

    double logL = 0.0;
    double logPrior = 0.0;
    double logPost = 0.0;
    double aic = 0.0;
    double bic = 0.0;

    int status = -1;

    std::unordered_map<std::string, std::vector<double>> params;
    std::unordered_map<std::string, std::vector<double>> param_sd;
};

struct OutputFitSlot {
    std::map<std::string, std::vector<OutputFitRow>> by_condition;
};

struct OutputEstimatorSlot {
    std::string name;
    std::string backend;
    std::string algorithm;
    std::string global_algorithm;
    std::string local_algorithm;

    std::unordered_map<std::string, double> numeric_control;
    std::unordered_map<std::string, std::string> string_control;
    std::unordered_map<std::string, bool> bool_control;
    std::unordered_map<std::string, std::vector<double>> vector_control;
};

struct OutputDiagnosticsRow {
    double subid = 0.0;
    std::string cond;

    int status = -1;
    int n_evals = 0;
    int result_code = -1;
    int n_draws = 0;
    int n_chains = 0;
    int warmup = 0;
    int thin = 1;
    int leapfrog_steps = 0;

    double optimum_value = 0.0;
    double accept_rate = 0.0;
    double step_size = 0.0;

    std::string result_message;
    std::string stop_reason;

    std::vector<double> draw_logPost;
    std::unordered_map<std::string, std::vector<std::vector<double>>> samples;
};

struct OutputDiagnosticsSlot {
    std::map<std::string, std::vector<OutputDiagnosticsRow>> by_condition;

    int em_iterations = 0;
    std::unordered_map<std::string, int> em_iterations_by_cond;
    std::unordered_map<std::string, std::string> em_stop_reason_by_cond;
};

struct EstimateOutput {
    OutputFitSlot fit;
    OutputEstimatorSlot estimator;
    OutputDiagnosticsSlot diagnostics;
};

namespace modify_outputs {

std::vector<std::string> ordered_base_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::vector<double>>& best_params
);

std::vector<std::string> ordered_flat_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::size_t>& param_sizes,
    const std::unordered_map<std::string, std::vector<double>>& best_params
);

SubjectMCMCResult summarize_mcmc_result(
    const SubjectFitTask& task,
    const std::vector<HMCSamplerResult>& chain_results,
    const StanControl& control,
    int n_evals
);

SubjectMCMCResult failed_mcmc_result(
    const SubjectFitTask& task,
    const StanControl& control,
    const std::string& message,
    const std::string& stop_reason
);

OutputFitSlot NLopt_fit(
    const std::vector<SubjectFitResult>& results
);

OutputEstimatorSlot NLopt_estimator(
    const std::string& estimator_name,
    const NLoptControl& control
);

OutputDiagnosticsSlot NLopt_diagnostics(
    const std::vector<SubjectFitResult>& results,
    int em_iterations = 0,
    const std::unordered_map<std::string, int>& em_iterations_by_cond = {},
    const std::unordered_map<std::string, std::string>&
        em_stop_reason_by_cond = {}
);

OutputFitSlot Stan_fit(
    const std::vector<SubjectMCMCResult>& results
);

OutputEstimatorSlot Stan_estimator(
    const std::string& estimator_name,
    const StanControl& control
);

OutputDiagnosticsSlot Stan_diagnostics(
    const std::vector<SubjectMCMCResult>& results
);

EstimateOutput combine_output_slots(
    const OutputFitSlot& fit,
    const OutputEstimatorSlot& estimator,
    const OutputDiagnosticsSlot& diagnostics
);

} // namespace modify_outputs
