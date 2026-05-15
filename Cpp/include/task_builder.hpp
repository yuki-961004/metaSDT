#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "criterion_prior.hpp"
#include "matrix_freq.hpp"
#include "modify_params.hpp"
#include "modify_prior.hpp"

/* ========================================================================== *
 *                            Subject Fit Task                                *
 * ========================================================================== */

// 每个任务像一个已经打包好的工作单, 后端算法只需要打开它并计算.
struct SubjectFitTask {
    double subid;
    std::string cond;
    std::string model;
    MatrixFreq freq;
    ModifiedParamsResult params;
    CriterionPrior prior;
};

/* ========================================================================== *
 *                              Task Builder                                  *
 * ========================================================================== */

// Build independent per-subject fitting tasks.
// 这里只准备数据和参数, 不关心之后由 NLopt 还是 MCMC 来求解.
std::vector<SubjectFitTask> build_fit_tasks(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower
        = {},
    const std::unordered_map<std::string, std::vector<double>>& custom_upper
        = {},
    bool apply_priors = false,
    const std::unordered_map<std::string, UserPrior>& user_priors = {}
);
