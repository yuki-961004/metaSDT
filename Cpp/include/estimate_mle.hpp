#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "build_objective.hpp" // 引入任务包与 nll 函数

// NLOPT 优化器的超参数配置结构体
struct NLoptControl {
    std::string algorithm = "LN_BOBYQA";
    double xtol_rel = 1e-5;
    int maxeval = 3000;
    double ftol_rel = 0.0;
    double ftol_abs = 0.0;
    double xtol_abs = 0.0;
    double maxtime = 0.0;
    double stopval = 0.0;
    int population = 0;
    double initial_step = 0.0;
    std::string local_algorithm = "LN_BOBYQA"; // 专供 MLSL/AUGLAG 使用的附属局部算法
};

// 定义返回给外层的单个被试拟合结果
struct SubjectFitResult {
    double subid = 0.0;
    std::string cond;
    double logL = 0.0; // 最大对数似然
    double aic = 0.0;  // 赤池信息量准则
    double bic = 0.0;  // 贝叶斯信息量准则
    std::unordered_map<std::string, std::vector<double>> best_params; // 最佳参数组合 (包含 fixed 和 constant)
    int status = -1;  // NLOPT 优化结束的状态码 (如 1 代表成功收敛)
};

// 暴露给外层的极大似然估计主函数
std::vector<SubjectFitResult> estimate_mle(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const NLoptControl& control,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper
);