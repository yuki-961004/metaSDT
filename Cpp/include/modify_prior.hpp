#ifndef MODIFY_PRIOR_HPP
#define MODIFY_PRIOR_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include "criterion_prior.hpp"
#include "modify_params.hpp" // 依赖于参数展平后的信息

// 定义用户从 Python/R 传来的单体先验结构
struct UserPrior {
    std::string type;
    // 允许用户使用任意统计学名称传递参数 (如 mean, sd, shape1, rate 等)
    std::unordered_map<std::string, double> args;
};

// 声明：获取模型全局默认先验配置 (MAP 估计或贝叶斯的默认信息)
std::unordered_map<std::string, UserPrior> default_priors();

// ==============================================================================
// 核心函数：先验参数标准化器
// ==============================================================================
// 它接收用户给出的命名先验字典，结合已经拍扁的 param_info，
// 自动计算每个参数在 Eigen 梯度向量中的一维绝对索引，并装载到 CriterionPrior 引擎中。
CriterionPrior modify_prior(
    const std::unordered_map<std::string, UserPrior>& user_priors,
    const ModifiedParamsResult& param_info,
    bool apply_priors = true
);

#endif // MODIFY_PRIOR_HPP