#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "estimate_mle.hpp" // 引入 SubjectFitResult 和 NLoptControl
#include "modify_prior.hpp" // 引入 UserPrior

// 暴露给外层的最大后验估计 (MAP) 主函数
std::vector<SubjectFitResult> estimate_map(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const NLoptControl& control,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper,
    const std::unordered_map<std::string, UserPrior>& user_priors,
    int* em_iterations_used = nullptr,
    std::unordered_map<std::string, int>* em_iterations_by_cond = nullptr,
    std::unordered_map<std::string, std::string>* em_stop_reason_by_cond = nullptr
);
