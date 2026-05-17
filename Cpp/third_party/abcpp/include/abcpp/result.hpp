#pragma once

#include "abcpp/matrix.hpp"
#include "abcpp/options.hpp"

#include <string>
#include <vector>

namespace abcpp {

struct ReductionInfo {
    ReductionMethod method = ReductionMethod::None;
    std::size_t ncomp = 0;
    Matrix rotation;
    std::vector<double> center;
};

struct AbcDiagnostics {
    double aic = 0.0;
    double bic = 0.0;
    std::vector<double> lambda;
};

struct AbcResult {
    Matrix adj_values;
    Matrix unadj_values;
    Matrix accepted_sumstats;
    Matrix weights;
    Matrix residuals;
    std::vector<double> distances;
    std::vector<std::size_t> accepted_indices;
    std::vector<bool> region;
    std::vector<bool> na_action;
    std::vector<Transform> transformations;
    Matrix logit_bounds;
    AbcOptions options;
    Method method = Method::Rejection;
    Kernel kernel = Kernel::Epanechnikov;
    bool hcorr = true;
    std::vector<double> lambda;
    int numparam = 0;
    int numstat = 0;
    double aic = 0.0;
    double bic = 0.0;
    ReductionInfo reduction;
    std::string status = "ok";
    std::string message;
    AbcDiagnostics diagnostics;
    std::vector<std::string> parameter_names;
    std::vector<std::string> statistic_names;
};

struct SummaryColumn {
    double min = 0.0;
    double q_lower = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double mode = 0.0;
    double q_upper = 0.0;
    double max = 0.0;
    double sd = 0.0;
};

struct SummaryResult {
    std::vector<SummaryColumn> columns;
    double interval = 0.95;
    bool unadjusted = false;
};

}  // namespace abcpp
