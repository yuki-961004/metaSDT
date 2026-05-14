#include "../include/algorithm_nlopt.hpp"

#include <stdexcept>
#include <string>

/* ========================================================================== *
 *                      NLopt Optimizer Construction Helpers                   *
 * ========================================================================== */

void sanitize_initial_point(
    std::vector<double>& x0,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double epsilon
) {
    for (size_t j = 0; j < x0.size(); ++j) {
        if (x0[j] <= lower_bounds[j]) {
            x0[j] = lower_bounds[j] + epsilon;
        }
        if (x0[j] >= upper_bounds[j]) {
            x0[j] = upper_bounds[j] - epsilon;
        }
    }
}

nlopt::opt create_nlopt_optimizer(
    const NLoptControl& control,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds
) {
    const unsigned int n_params = static_cast<unsigned int>(
        lower_bounds.size()
    );

    nlopt::opt opt(control.algorithm.c_str(), n_params);
    opt.set_lower_bounds(lower_bounds);
    opt.set_upper_bounds(upper_bounds);

    if (control.algorithm.find("MLSL") != std::string::npos ||
        control.algorithm.find("AUGLAG") != std::string::npos) {
        nlopt::opt local_opt(control.local_algorithm.c_str(), n_params);
        local_opt.set_xtol_rel(control.xtol_rel);

        if (control.ftol_rel > 0) {
            local_opt.set_ftol_rel(control.ftol_rel);
        }
        if (control.ftol_abs > 0) {
            local_opt.set_ftol_abs(control.ftol_abs);
        }
        if (control.xtol_abs > 0) {
            local_opt.set_xtol_abs(control.xtol_abs);
        }

        opt.set_local_optimizer(local_opt);
    }

    opt.set_xtol_rel(control.xtol_rel);
    opt.set_maxeval(control.maxeval);

    if (control.ftol_rel > 0) {
        opt.set_ftol_rel(control.ftol_rel);
    }
    if (control.ftol_abs > 0) {
        opt.set_ftol_abs(control.ftol_abs);
    }
    if (control.xtol_abs > 0) {
        opt.set_xtol_abs(control.xtol_abs);
    }
    if (control.maxtime > 0) {
        opt.set_maxtime(control.maxtime);
    }
    if (control.population > 0) {
        opt.set_population(control.population);
    }
    if (control.initial_step > 0) {
        opt.set_initial_step(control.initial_step);
    }
    if (control.stopval != 0.0) {
        opt.set_stopval(control.stopval);
    }

    if (!control.x_weights.empty()) {
        if (control.x_weights.size() != lower_bounds.size()) {
            throw std::invalid_argument(
                "Error: control.x_weights length must equal number of "
                "free parameters."
            );
        }
        opt.set_x_weights(control.x_weights);
    }

    if (control.vector_storage > 0) {
        opt.set_vector_storage(control.vector_storage);
    }

    for (const auto& kv : control.nlopt_params) {
        opt.set_param(kv.first.c_str(), kv.second);
    }

    return opt;
}
