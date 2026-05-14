#include "../include/modify_prior.hpp"
#include <algorithm>
#include <stdexcept>

namespace {

/* ========================================================================== *
 *                              Internal Helpers                              *
 * ========================================================================== */

// 将文本转换为小写，以便进行不区分大小写的类型匹配.
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

} // namespace

/* ========================================================================== *
 *                           Default Prior Settings                           *
 * ========================================================================== */

std::unordered_map<std::string, UserPrior> default_priors() {
    std::unordered_map<std::string, UserPrior> priors;

    // 当用户未指定时使用的核心 SDT 默认先验.
    priors["d"] = {"normal", {{"mean", 1.5}, {"sd", 3.0}}};
    priors["c_resp"] = {"normal", {{"mean", 0.0}, {"sd", 2.0}}};
    priors["c_conf"] = {"normal", {{"mean", 0.0}, {"sd", 2.0}}};
    priors["lapse"] = {"beta", {{"shape1", 1.5}, {"shape2", 18.5}}};

    return priors;
}

/* ========================================================================== *
 *                     Build Criterion Prior from User Config                 *
 * ========================================================================== */

CriterionPrior modify_prior(
    const std::unordered_map<std::string, UserPrior>& user_priors,
    const ModifiedParamsResult& param_info,
    bool apply_priors
) {
    CriterionPrior criterion_prior;

    // MLE 路径：不应应用任何先验.
    if (!apply_priors) {
        return criterion_prior;
    }

/* ========================================================================== *
 *               1) Merge default priors with user overrides                  *
 * ========================================================================== */

    // 用户配置的优先级高于默认配置.
    auto merged_priors = default_priors();
    for (const auto& kv : user_priors) {
        merged_priors[kv.first] = kv.second;
    }

/* ========================================================================== *
 *       2) Walk free parameters and map to flattened parameter index         *
 * ========================================================================== */

    int flat_index = 0;

    // 顺序必须与 name_free 严格一致，以与优化器向量对齐.
    for (const auto& param_name : param_info.name_free) {

        // 每个自由参数可以是标量或向量值.
        size_t p_size = param_info.structured.free.at(param_name).size();

        // 仅当此参数具有配置的先验条目时，才应用先验.
        if (merged_priors.count(param_name)) {
            const auto& up = merged_priors.at(param_name);
            std::string type_str = to_lower(up.type);

            CriterionPrior::PriorType p_type = CriterionPrior::PriorType::NONE;

            // 接受别名，使得用户输入更加稳健.
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

            // 仅当先验类型不为 NONE 时，才将其注册.
            if (p_type != CriterionPrior::PriorType::NONE) {
                double p1 = 0.0;
                double p2 = 0.0;

                // 从候选别名中提取一个参数.
                // 如果未提供任何候选参数，则抛出明确的错误.
                auto extract_arg = [&](const std::vector<std::string>& keys) {
                    for (const auto& k : keys) {
                        if (up.args.count(k)) {
                            return up.args.at(k);
                        }
                    }
                    std::string err = "Error: Missing required prior argument"
                                      " for '" + param_name + "'. Expected: ";
                    for (size_t k = 0; k < keys.size(); ++k) {
                        err += "'" + keys[k] + "'";
                        if (k < keys.size() - 1) {
                            err += ", ";
                        }
                    }
                    throw std::invalid_argument(err);
                };

                // 将分布类型映射为规范的参数名称.
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
                    p2 = 0.0;
                }

                // 默认情况下，向量参数的所有元素共享相同的先验配置.
                for (size_t i = 0; i < p_size; ++i) {
                    criterion_prior.add_prior(flat_index + i, p_type, p1, p2);
                }
            }
        }

        // 始终推进扁平索引，以保持后续参数的对齐.
        flat_index += p_size;
    }

    return criterion_prior;
}
