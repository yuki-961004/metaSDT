#pragma once

#include "task_builder.hpp"
#include "modify_prior.hpp"

#include <string>
#include <unordered_map>
#include <vector>

struct ABCNnetControl {
    int numnet = 0;
    int sizenet = 0;
    std::vector<double> lambda;
    int maxit = 0;
    double rang = 0.0;
    double abstol = 0.0;
    double reltol = 0.0;
    bool verbose = false;
    bool skip = false;
};

struct ABCControl {
    double tol = 0.0;
    std::string method;
    std::string reduction;
    int n_comp = 0;
    int samples = 0;
    std::string kernel;
    bool hcorr = true;
    std::vector<std::string> transf;
    std::vector<std::vector<double>> logit_bounds;
    std::vector<bool> subset;
    std::vector<double> prior_weights;
    unsigned int seed = 1004;
    ABCNnetControl nnet;
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
    std::vector<double> accepted_weights;
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
