#include "../include/modify_prior.hpp"
#include <algorithm>
#include <stdexcept>

namespace {
    // 辅助函数：字符串转小写，实现容错匹配
    std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    }
}

std::unordered_map<std::string, UserPrior> default_priors() {
    std::unordered_map<std::string, UserPrior> priors;
    
    // Default priors based on SDT theory (MAP/Bayesian estimation):
    // d' (sensitivity): Normal(1.5, 3.0)
    // c_resp (criterion): Normal(0.0, 2.0)
    // c_conf (meta-criteria): Normal(0.0, 2.0)
    priors["d"] = {"normal", {{"mean", 1.5}, {"sd", 3.0}}}; 
    priors["c_resp"] = {"normal", {{"mean", 0.0}, {"sd", 2.0}}};
    priors["c_conf"] = {"normal", {{"mean", 0.0}, {"sd", 2.0}}};
    priors["lapse"] = {"beta", {{"shape1", 1.5}, {"shape2", 18.5}}};
    
    return priors;
}

CriterionPrior modify_prior(
    const std::unordered_map<std::string, UserPrior>& user_priors,
    const ModifiedParamsResult& param_info,
    bool apply_priors
) {
    CriterionPrior criterion_prior;

    if (!apply_priors) {
        // Return an empty prior evaluator for MLE
        return criterion_prior;
    }

    // 1. 合并默认先验与用户输入的先验 (以用户输入为最高优先级)
    auto merged_priors = default_priors();
    for (const auto& kv : user_priors) {
        merged_priors[kv.first] = kv.second;
    }

    // 2. 遍历自由参数列表，计算在扁平梯度向量 theta 中的绝对索引
    int flat_index = 0;
    
    // 我们必须严格按照 name_free 的顺序推演索引，这与 nll 函数和 LogPosterior 完全一致！
    for (const auto& param_name : param_info.name_free) {
        
        // 获取该参数占据的向量长度 (例如标量 d 为 1, c_conf 可能是 3)
        size_t p_size = param_info.structured.free.at(param_name).size();

        // 如果用户为这个参数配置了先验
        if (merged_priors.count(param_name)) {
            const auto& up = merged_priors.at(param_name);
            std::string type_str = to_lower(up.type);
            
            CriterionPrior::PriorType p_type = CriterionPrior::PriorType::NONE;
            
            // 智能容错解析
            if (type_str == "normal" || type_str == "norm") {
                p_type = CriterionPrior::PriorType::NORMAL;
            } else if (type_str == "uniform" || type_str == "unif") {
                p_type = CriterionPrior::PriorType::UNIFORM;
            } else if (type_str == "lognormal" || type_str == "lnorm") {
                p_type = CriterionPrior::PriorType::LOGNORMAL;
            } else if (type_str == "cauchy") {
                p_type = CriterionPrior::PriorType::CAUCHY;
            } else if (type_str == "beta") {
                p_type = CriterionPrior::PriorType::BETA;
            } else if (type_str == "exponential" || type_str == "exp") {
                p_type = CriterionPrior::PriorType::EXPONENTIAL;
            } else if (type_str == "none") {
                p_type = CriterionPrior::PriorType::NONE;
            } else {
                throw std::invalid_argument(
                    "Error: Unknown prior type '" + up.type + 
                    "' for parameter '" + param_name + "'."
                );
            }

            // 若存在有效分布，向后验求值器注入该分布配置
            if (p_type != CriterionPrior::PriorType::NONE) {
                double p1 = 0.0, p2 = 0.0;
                
                // 智能提取闭包：遍历候选用词，若没找到则精准向用户报错
                auto extract_arg = [&](const std::vector<std::string>& keys) {
                    for (const auto& k : keys) {
                        if (up.args.count(k)) return up.args.at(k);
                    }
                    std::string err = "Error: Missing required prior argument"
                                      " for '" + param_name + "'. Expected: ";
                    for (size_t k = 0; k < keys.size(); ++k) {
                        err += "'" + keys[k] + "'";
                        if (k < keys.size() - 1) err += ", ";
                    }
                    throw std::invalid_argument(err);
                };

                // 动态容错映射 (Smart Parameter Mapping)
                if (p_type == CriterionPrior::PriorType::NORMAL || 
                    p_type == CriterionPrior::PriorType::LOGNORMAL || 
                    p_type == CriterionPrior::PriorType::CAUCHY) {
                    p1 = extract_arg({"mean", "mu", "location", "param1"});
                    p2 = extract_arg({"sd", "sigma", "scale", "param2"});
                } else if (p_type == CriterionPrior::PriorType::UNIFORM) {
                    p1 = extract_arg({"min", "lower", "param1"});
                    p2 = extract_arg({"max", "upper", "param2"});
                } else if (p_type == CriterionPrior::PriorType::BETA) {
                    p1 = extract_arg({"shape1", "alpha", "param1"});
                    p2 = extract_arg({"shape2", "beta", "param2"});
                } else if (p_type == CriterionPrior::PriorType::EXPONENTIAL) {
                    p1 = extract_arg({"rate", "lambda", "param1"});
                    p2 = 0.0; // 指数分布只需一个参数，置空占位
                }

                for (size_t i = 0; i < p_size; ++i) {
                    // 如果是像 c_conf 这样的数组，默认其所有元素共享该先验分布
                    criterion_prior.add_prior(flat_index + i, p_type, p1, p2);
                }
            }
        }
        // 无论该参数是否有先验，都必须推进扁平索引带，保证后续参数的索引能够正确对齐！
        flat_index += p_size;
    }
    return criterion_prior;
}