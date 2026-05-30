#pragma once

#include <metaSDT/estimate_abc.hpp>
#include <metaSDT/modify_control.hpp>
#include <metaSDT/task_builder.hpp>

#include <unordered_map>
#include <string>

namespace abcppAdapter {

// 将用户自定义的先验配置与默认核心先验进行合并
std::unordered_map<std::string, UserPrior> merged_priors(
    const std::unordered_map<std::string, UserPrior>& user_priors
);

// 执行单个被试的近似贝叶斯计算 (ABC) 拟合任务
SubjectABCResult run_subject_abc(
    const SubjectFitTask& task,
    const ABCControl& control,
    const std::unordered_map<std::string, UserPrior>& prior_map,
    int task_index
);

} // namespace abcppAdapter