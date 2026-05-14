#include "../include/modify_prior.hpp"
#include <algorithm>
#include <stdexcept>

namespace {

/* ========================================================================== *
 *                              Internal Helpers                               *
 * ========================================================================== */

// Convert text to lowercase so type matching is case-insensitive.
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

} // namespace

/* ========================================================================== *
 *                           Default Prior Settings                            *
 * ========================================================================== */

std::unordered_map<std::string, UserPrior> default_priors() {
    std::unordered_map<std::string, UserPrior> priors;

    // Core SDT priors used when user does not override.
    priors["d"] = {"normal", {{"mean", 1.5}, {"sd", 3.0}}};
    priors["c_resp"] = {"normal", {{"mean", 0.0}, {"sd", 2.0}}};
    priors["c_conf"] = {"normal", {{"mean", 0.0}, {"sd", 2.0}}};
    priors["lapse"] = {"beta", {{"shape1", 1.5}, {"shape2", 18.5}}};

    return priors;
}

/* ========================================================================== *
 *                     Build Criterion Prior from User Config                  *
 * ========================================================================== */

CriterionPrior modify_prior(
    const std::unordered_map<std::string, UserPrior>& user_priors,
    const ModifiedParamsResult& param_info,
    bool apply_priors
) {
    CriterionPrior criterion_prior;

    // MLE path: no prior should be applied.
    if (!apply_priors) {
        return criterion_prior;
    }

    /* ---------------------------------------------------------------------- *
     *              1) Merge default priors with user overrides               *
     * ---------------------------------------------------------------------- */

    // User config has higher priority than defaults.
    auto merged_priors = default_priors();
    for (const auto& kv : user_priors) {
        merged_priors[kv.first] = kv.second;
    }

    /* ---------------------------------------------------------------------- *
     *       2) Walk free parameters and map to flattened parameter index      *
     * ---------------------------------------------------------------------- */

    int flat_index = 0;

    // Order must follow name_free exactly to align with optimizer vector.
    for (const auto& param_name : param_info.name_free) {

        // Each free parameter may be scalar or vector-valued.
        size_t p_size = param_info.structured.free.at(param_name).size();

        // Apply prior only when this parameter has a configured prior entry.
        if (merged_priors.count(param_name)) {
            const auto& up = merged_priors.at(param_name);
            std::string type_str = to_lower(up.type);

            CriterionPrior::PriorType p_type = CriterionPrior::PriorType::NONE;

            // Accept aliases so user input is more robust.
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

            // Register prior only when type is not NONE.
            if (p_type != CriterionPrior::PriorType::NONE) {
                double p1 = 0.0;
                double p2 = 0.0;

                // Extract one argument from candidate aliases.
                // Throw a clear error if none of them is provided.
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

                // Map distribution type to canonical parameter names.
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

                // Vector parameters share the same prior config by default.
                for (size_t i = 0; i < p_size; ++i) {
                    criterion_prior.add_prior(flat_index + i, p_type, p1, p2);
                }
            }
        }

        // Always advance flattened index to keep later parameters aligned.
        flat_index += p_size;
    }

    return criterion_prior;
}
