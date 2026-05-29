#ifndef MODIFY_PRIORS_HPP
#define MODIFY_PRIORS_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include "criterion_prior.hpp"
#include "modify_params.hpp"

/* ========================================================================== *
 *                              Prior Settings                                *
 * ========================================================================== */

struct UserPrior {
    std::string type;
    std::unordered_map<std::string, double> args;
};

std::unordered_map<std::string, UserPrior> default_priors();

/* ========================================================================== *
 *                         Prior Standardization                              *
 * ========================================================================== */

CriterionPrior modify_priors(
    const std::unordered_map<std::string, UserPrior>& user_priors,
    const ModifiedParamsResult& param_info,
    bool apply_priors = true
);

#endif // MODIFY_PRIORS_HPP
