#pragma once

#include "task_builder.hpp"
#include "modify_prior.hpp"

#include <string>
#include <unordered_map>
#include <vector>

struct ABCControl {
    double tol = 0.1;
    std::string method = "rejection";
    std::string reduction = "none";
    int n_comp = 0;
    int samples = 1000;
    std::string kernel = "epanechnikov";
    bool hcorr = true;
    unsigned int seed = 1004;
    int print_level = 1;
};

struct ABCSummaryStats {
    double min = 0.0;
    double q_lower = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double mode = 0.0;
    double q_upper = 0.0;
    double max = 0.0;
    double sd = 0.0;
};

struct SubjectABCResult {
    double subid = 0.0;
    std::string cond;
    std::vector<std::string> parameter_names;
    std::vector<ABCSummaryStats> summary;
    std::vector<double> accepted_distances;
    std::vector<std::size_t> accepted_indices;
    int n_comp_used = 0;
    int status = 0;
    std::string message;
};

std::vector<SubjectABCResult> estimate_abc(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const ABCControl& control,
    const std::unordered_map<std::string, UserPrior>& user_priors = {}
);
