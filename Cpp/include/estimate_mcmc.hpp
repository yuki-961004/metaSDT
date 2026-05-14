#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "modify_control.hpp"
#include "modify_outputs.hpp"
#include "modify_params.hpp"
#include "modify_prior.hpp"

/* ========================================================================== *
 *                              Main Public API                               *
 * ========================================================================== */

// 执行基于 Stan Math 后验梯度的 HMC-MCMC 估计入口点
std::vector<SubjectMCMCResult> estimate_mcmc(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const StanControl& control,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper,
    const std::unordered_map<std::string, UserPrior>& user_priors
);
