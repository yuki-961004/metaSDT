#include <metaSDT/modify_priors.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace {

/* ========================================================================== *
 *                              Internal Helpers                              *
 * ========================================================================== */

std::string to_lower(std::string text) {
    // 将先验类型统一转成小写, 这样用户输入 normal/NORMAL 都能匹配.
    std::transform(
        text.begin(),
        text.end(),
        text.begin(),
        [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        }
    );

    return text;
}

} // namespace

/* ========================================================================== *
 *                           Default Prior Settings                           *
 * ========================================================================== */

std::unordered_map<std::string, UserPrior> default_priors() {
    std::unordered_map<std::string, UserPrior> priors;

    // 当用户没有覆盖时, 使用项目默认的核心 SDT 先验配置.
    priors["d"] = {"normal", {{"mean", 1.5}, {"sd", 3.0}}};
    priors["c_resp"] = {"normal", {{"mean", 0.0}, {"sd", 2.0}}};
    priors["c_conf"] = {"normal", {{"mean", 0.0}, {"sd", 2.0}}};
    priors["lapse"] = {"beta", {{"shape1", 1.5}, {"shape2", 18.5}}};

    return priors;
}

/* ========================================================================== *
 *                     Build Criterion Prior from User Config                 *
 * ========================================================================== */

CriterionPrior modify_priors(
    const std::unordered_map<std::string, UserPrior>& user_priors,
    const ModifiedParamsResult& param_info,
    bool apply_priors
) {
    CriterionPrior criterion_prior;

    // MLE 路径不应该应用任何先验, 所以直接返回空先验引擎.
    if (!apply_priors) {
        return criterion_prior;
    }

    /* ====================================================================== *
     *                Merge default priors with user overrides                *
     * ====================================================================== */

    // 用户配置的优先级高于默认配置.
    std::unordered_map<std::string, UserPrior> merged_priors =
        default_priors();
    for (const auto& kv : user_priors) {
        merged_priors[kv.first] = kv.second;
    }

    /* ====================================================================== *
     *           Walk free parameters and map to flattened indices            *
     * ====================================================================== */

    int flat_index = 0;

    // name_free 的顺序必须和优化向量一致, 所以逐个参数累加扁平索引.
    for (const auto& param_name : param_info.name_free) {
        const std::vector<double>& values =
            param_info.structured.free.at(param_name);
        const std::size_t param_size = values.size();

        // 只有参数存在先验条目时才解析并注册先验.
        if (merged_priors.count(param_name)) {
            const UserPrior& user_prior = merged_priors.at(param_name);
            const std::string type_text = to_lower(user_prior.type);

            CriterionPrior::PriorType prior_type =
                CriterionPrior::PriorType::NONE;

            // 接受常见别名, 让外部 wrapper 的输入更稳健.
            if (type_text == "normal" || type_text == "norm") {
                prior_type = CriterionPrior::PriorType::NORMAL;
            } else if (type_text == "uniform" || type_text == "unif") {
                prior_type = CriterionPrior::PriorType::UNIFORM;
            } else if (type_text == "lognormal" || type_text == "lnorm") {
                prior_type = CriterionPrior::PriorType::LOGNORMAL;
            } else if (type_text == "cauchy") {
                prior_type = CriterionPrior::PriorType::CAUCHY;
            } else if (type_text == "beta") {
                prior_type = CriterionPrior::PriorType::BETA;
            } else if (type_text == "exponential" || type_text == "exp") {
                prior_type = CriterionPrior::PriorType::EXPONENTIAL;
            } else if (type_text == "none") {
                prior_type = CriterionPrior::PriorType::NONE;
            } else {
                throw std::invalid_argument(
                    "Error: Unknown prior type '" + user_prior.type +
                    "' for parameter '" + param_name + "'."
                );
            }

            // NONE 表示用户显式关闭该参数先验, 不需要注册到引擎里.
            if (prior_type != CriterionPrior::PriorType::NONE) {
                double param1 = 0.0;
                double param2 = 0.0;

                auto extract_arg = [
                    &user_prior,
                    &param_name
                ](const std::vector<std::string>& keys) {
                    // 依次尝试别名, 允许用户用 mean/mu/param1 等名称.
                    for (const auto& key : keys) {
                        if (user_prior.args.count(key)) {
                            return user_prior.args.at(key);
                        }
                    }

                    std::string error =
                        "Error: Missing required prior argument for '" +
                        param_name + "'. Expected: ";
                    for (std::size_t index = 0; index < keys.size(); ++index) {
                        error += "'" + keys[index] + "'";
                        if (index < keys.size() - 1) {
                            error += ", ";
                        }
                    }

                    throw std::invalid_argument(error);
                };

                // 不同分布需要不同的两个参数槽位.
                if (prior_type == CriterionPrior::PriorType::NORMAL ||
                    prior_type == CriterionPrior::PriorType::LOGNORMAL ||
                    prior_type == CriterionPrior::PriorType::CAUCHY) {
                    param1 = extract_arg({"mean", "mu", "location", "param1"});
                    param2 = extract_arg({"sd", "sigma", "scale", "param2"});
                } else if (prior_type == CriterionPrior::PriorType::UNIFORM) {
                    param1 = extract_arg({"min", "lower", "param1"});
                    param2 = extract_arg({"max", "upper", "param2"});
                } else if (prior_type == CriterionPrior::PriorType::BETA) {
                    param1 = extract_arg({"shape1", "alpha", "param1"});
                    param2 = extract_arg({"shape2", "beta", "param2"});
                } else if (
                    prior_type == CriterionPrior::PriorType::EXPONENTIAL
                ) {
                    param1 = extract_arg({"rate", "lambda", "param1"});
                    param2 = 0.0;
                }

                // 向量参数的每个元素共享同一个先验配置.
                for (std::size_t index = 0; index < param_size; ++index) {
                    criterion_prior.add_prior(
                        flat_index + static_cast<int>(index),
                        prior_type,
                        param1,
                        param2
                    );
                }
            }
        }

        // 无论该参数是否有先验, 都要推进扁平索引保持后续对齐.
        flat_index += static_cast<int>(param_size);
    }

    return criterion_prior;
}
