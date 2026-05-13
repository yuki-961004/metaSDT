#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "build_objective.hpp"

struct NLoptControl {
    std::string algorithm = "LN_BOBYQA";
    double xtol_rel = 1e-6;
    int maxeval = 10000;
    double ftol_rel = 0.0;
    double ftol_abs = 0.0;
    double xtol_abs = 0.0;
    double maxtime = 0.0;
    double stopval = 0.0;
    int population = 0;
    double initial_step = 0.0;
    std::string local_algorithm = "LN_BOBYQA";
    std::vector<double> x_weights;
    long seed = 1004;
    std::unordered_map<std::string, double> nlopt_params;
    int vector_storage = 0; // 0 means NLopt heuristic default.
    int print_level = 1;
    int threads = 0; // <=0 means use all available threads.
    std::string progress = "dynamic"; // auto | dynamic | line | silent
    int progress_refresh_ms = 100;
    double progress_line_interval_sec = 2.0;
    double progress_line_interval_pct = 5.0;

    // EM-MAP controls (used by estimate_map only).
    int em_max_iter = 100;
    double em_tol = 1e-3;
    int em_patience = 0; // 0 means disabled.
    bool em_init_mle = true;
};

struct SubjectFitResult {
    double subid = 0.0;
    std::string cond;
    double logL = 0.0;
    double logPrior = 0.0;
    double logPost = 0.0;
    double aic = 0.0;
    double bic = 0.0;
    std::unordered_map<std::string, std::vector<double>> best_params;
    int status = -1;
    int n_evals = 0;
    double optimum_value = 0.0;
    std::string result_message;
    std::string stop_reason;
};

std::vector<SubjectFitResult> estimate_mle(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const NLoptControl& control,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper
);
