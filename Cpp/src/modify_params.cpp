#include "../include/modify_params.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

/* ========================================================================== *
 *                           Default Parameter Values                          *
 * ========================================================================== */

// Build default parameter groups for all supported models.
ParamGroup default_params() {
    ParamGroup defaults;

    /* ---------------------------------------------------------------------- *
     *                           Free Parameters                               *
     * ---------------------------------------------------------------------- */

    defaults.free["d"] = {1.5};
    defaults.free["c_resp"] = {0.0};

    /* ---------------------------------------------------------------------- *
     *                          Fixed Parameters                               *
     * ---------------------------------------------------------------------- */

    // Type-1 / sensory parameters.
    defaults.fixed["sd_signal"] = {1.0};
    defaults.fixed["sd_noise"] = {1.0};

    // Type-2 / confidence parameters.
    defaults.fixed["c_conf"] = {};
    defaults.fixed["n_conf"] = {};

    // Model-specific extensions.
    defaults.fixed["sigma_meta"] = {0.5};
    defaults.fixed["meta_uncertainty"] = {0.5};
    defaults.fixed["sigma_c"] = {0.1};
    defaults.fixed["rho_decay"] = {0.5};
    defaults.fixed["delta_post"] = {0.5};
    defaults.fixed["tau"] = {0.5};
    defaults.fixed["w_pe"] = {0.5};
    defaults.fixed["w_v"] = {0.5};
    defaults.fixed["w_u"] = {0.5};
    defaults.fixed["rho"] = {0.5};

    /* ---------------------------------------------------------------------- *
     *                         Constant Parameters                             *
     * ---------------------------------------------------------------------- */

    defaults.constant["sim_trials"] = {100000.0};
    defaults.constant["rate_lapse"] = {0.0};
    defaults.constant["calc_tol"] = {1e-10};
    defaults.constant["L"] = {};
    defaults.constant["penalty"] = {1.0};
    defaults.constant["rng_seed"] = {1004.0};

    // Sort switch for d:
    //  >0 ascending, <0 descending, 0 keep optimizer natural order.
    defaults.constant["sort_d"] = {0.0};

    return defaults;
}

/* ========================================================================== *
 *                             Default Bounds Map                              *
 * ========================================================================== */

// Build default lower/upper bounds for each parameter.
std::unordered_map<std::string, std::vector<double>> default_bounds() {
    std::unordered_map<std::string, std::vector<double>> bounds;

    // Core sensory parameters.
    bounds["d"] = {-1.0, 10.0};
    bounds["c_resp"] = {-5.0, 5.0};
    bounds["sd_signal"] = {1e-4, 5.0};
    bounds["sd_noise"] = {1e-4, 5.0};

    // Confidence boundaries.
    bounds["c_conf"] = {1e-4, 20.0};

    // Model-specific bounds.
    bounds["sigma_meta"] = {1e-4, 5.0};
    bounds["meta_uncertainty"] = {1e-4, 5.0};
    bounds["sigma_c"] = {1e-4, 5.0};
    bounds["rho_decay"] = {1e-4, 0.9999};
    bounds["w_pe"] = {1e-4, 1.0};
    bounds["w_u"] = {1e-4, 1.0};
    bounds["delta_post"] = {1e-4, 5.0};
    bounds["tau"] = {1e-4, 5.0};
    bounds["w_v"] = {1e-4, 5.0};
    bounds["rho"] = {-0.999, 0.999};

    return bounds;
}

/* ========================================================================== *
 *                         Core Parameter Normalization                        *
 * ========================================================================== */

ModifiedParamsResult modify_params(
    const ParamGroup& user_params,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper
) {

    /* ---------------------------------------------------------------------- *
     *                 1) Init defaults and free-parameter demotion            *
     * ---------------------------------------------------------------------- */

    ParamGroup params = default_params();

    // If user explicitly provides a free set, default free parameters that are
    // not listed are demoted to fixed so optimization scope is explicit.
    if (!user_params.free.empty()) {
        std::vector<std::string> to_demote;
        for (const auto& kv : params.free) {
            if (user_params.free.find(kv.first) == user_params.free.end()) {
                to_demote.push_back(kv.first);
            }
        }
        for (const auto& key : to_demote) {
            params.fixed[key] = params.free[key];
            params.free.erase(key);
        }
    }

    /* ---------------------------------------------------------------------- *
     *                  2) Overlay user inputs with mutual cleanup             *
     * ---------------------------------------------------------------------- */

    // Apply by priority: constant -> fixed -> free.
    // Each insert erases lower-priority duplicates from other groups.
    for (const auto& kv : user_params.constant) {
        params.constant[kv.first] = kv.second;
        params.free.erase(kv.first);
        params.fixed.erase(kv.first);
    }

    for (const auto& kv : user_params.fixed) {
        params.fixed[kv.first] = kv.second;
        params.free.erase(kv.first);
        params.constant.erase(kv.first);
    }

    for (const auto& kv : user_params.free) {
        params.free[kv.first] = kv.second;
        params.fixed.erase(kv.first);
        params.constant.erase(kv.first);
    }

    /* ---------------------------------------------------------------------- *
     *                     3) Flatten grouped params to one map                *
     * ---------------------------------------------------------------------- */

    std::unordered_map<std::string, std::vector<double>> flat_params;

    for (const auto& kv : params.constant) {
        flat_params[kv.first] = kv.second;
    }
    for (const auto& kv : params.fixed) {
        flat_params[kv.first] = kv.second;
    }
    for (const auto& kv : params.free) {
        flat_params[kv.first] = kv.second;
    }

    /* ---------------------------------------------------------------------- *
     *                  4) n_conf / c_conf / c_resp consistency                *
     * ---------------------------------------------------------------------- */

    auto it_n_conf = flat_params.find("n_conf");
    auto it_c_conf = flat_params.find("c_conf");

    bool has_n_conf = (it_n_conf != flat_params.end() &&
                       !it_n_conf->second.empty());
    bool has_c_conf = (it_c_conf != flat_params.end() &&
                       !it_c_conf->second.empty());

    if (!has_n_conf) {
        if (has_c_conf) {
            // Symmetric rule: n_conf = 2 * len(c_conf) + 1.
            double calc_n_conf = 2.0 * it_c_conf->second.size() + 1.0;
            flat_params["n_conf"] = {calc_n_conf};
            params.fixed["n_conf"] = {calc_n_conf};
        }
    } else {
        // If n_conf is given, c_conf is treated as full criterion vector.
        int n_conf_val = static_cast<int>(it_n_conf->second[0]);
        if (has_c_conf) {
            // For odd n_conf, enforce median(c_conf) as c_resp.
            if (n_conf_val % 2 != 0) {
                int mid_idx = n_conf_val / 2;
                if (mid_idx < static_cast<int>(it_c_conf->second.size())) {
                    double mid_val = it_c_conf->second[mid_idx];
                    auto it_c_resp = flat_params.find("c_resp");
                    if (it_c_resp != flat_params.end() &&
                        !it_c_resp->second.empty()) {
                        if (std::abs(it_c_resp->second[0] - mid_val) > 1e-6) {
                            std::cerr
                                << "Warning: You set c_resp and n_conf, "
                                << "but c_resp is not the median point "
                                << "of c_conf. Please check if c_resp is "
                                << "set correctly. The median point of "
                                << "c_conf (" << mid_val << ") will be "
                                << "used as c_resp for calculation.\n";
                        }
                    }

                    flat_params["c_resp"] = {mid_val};
                    if (params.free.count("c_resp")) {
                        params.free["c_resp"] = {mid_val};
                    }
                    if (params.fixed.count("c_resp")) {
                        params.fixed["c_resp"] = {mid_val};
                    }
                    if (params.constant.count("c_resp")) {
                        params.constant["c_resp"] = {mid_val};
                    }
                }
            }
        }
    }

    ModifiedParamsResult result;
    result.flat = flat_params;
    result.structured = params;

    // Detect whether c_conf is full absolute criterion vector.
    bool is_full_vector = false;
    if (has_n_conf && has_c_conf) {
        int n_conf_val = static_cast<int>(it_n_conf->second[0]);
        if (n_conf_val == static_cast<int>(it_c_conf->second.size())) {
            is_full_vector = true;
        }
    }

    /* ---------------------------------------------------------------------- *
     *                 5) Collect names/counts and build final bounds          *
     * ---------------------------------------------------------------------- */

    result.numb_free = 0;
    auto bounds_dict = default_bounds();

    // Sort keys alphabetically to ensure deterministic iteration order.
    auto sort_keys = [](std::vector<std::string>& keys) {
        std::sort(keys.begin(), keys.end());
    };

    // Free parameters: record names, counts, and per-element bounds.
    std::vector<std::string> keys_free;
    for (const auto& kv : params.free) {
        keys_free.push_back(kv.first);
    }
    sort_keys(keys_free);

    for (const auto& key : keys_free) {
        result.name_free.push_back(key);
        const auto& val_vec = params.free.at(key);
        size_t p_size = val_vec.size();
        result.numb_free += p_size;

        double lb_base = -1e5;
        double ub_base = 1e5;
        if (bounds_dict.count(key)) {
            lb_base = bounds_dict[key][0];
            ub_base = bounds_dict[key][1];
        }

        // Full-vector c_conf can cross zero, so use symmetric wide bounds.
        if (key == "c_conf" && is_full_vector) {
            lb_base = -10.0;
            ub_base = 10.0;
        }

        for (size_t i = 0; i < p_size; ++i) {
            double current_lb = lb_base;
            double current_ub = ub_base;

            if (custom_lower.count(key)) {
                if (custom_lower.at(key).size() > i) {
                    current_lb = custom_lower.at(key)[i];
                } else if (custom_lower.at(key).size() == 1) {
                    current_lb = custom_lower.at(key)[0];
                }
            }
            if (custom_upper.count(key)) {
                if (custom_upper.at(key).size() > i) {
                    current_ub = custom_upper.at(key)[i];
                } else if (custom_upper.at(key).size() == 1) {
                    current_ub = custom_upper.at(key)[0];
                }
            }

            result.lower_bounds.push_back(current_lb);
            result.upper_bounds.push_back(current_ub);
        }
    }

    // Fixed parameters: record names and element counts.
    result.numb_fixed = 0;
    std::vector<std::string> keys_fixed;
    for (const auto& kv : params.fixed) {
        keys_fixed.push_back(kv.first);
    }
    sort_keys(keys_fixed);
    for (const auto& key : keys_fixed) {
        result.name_fixed.push_back(key);
        result.numb_fixed += params.fixed.at(key).size();
    }

    // Constant parameters: record names and element counts.
    result.numb_constant = 0;
    std::vector<std::string> keys_constant;
    for (const auto& kv : params.constant) {
        keys_constant.push_back(kv.first);
    }
    sort_keys(keys_constant);
    for (const auto& key : keys_constant) {
        result.name_constant.push_back(key);
        result.numb_constant += params.constant.at(key).size();
    }

    return result;
}

/* ========================================================================== *
 *                    ModifiedParamsResult Member Functions                    *
 * ========================================================================== */

std::vector<double> ModifiedParamsResult::extract_free_vector(
    const std::unordered_map<std::string, std::vector<double>>& params_map
) const {
    std::vector<double> out;
    out.reserve(numb_free);

    // Append free parameters in the exact order stored in name_free.
    for (const auto& name : name_free) {
        const auto it = params_map.find(name);
        if (it != params_map.end()) {
            out.insert(out.end(), it->second.begin(), it->second.end());
        }
    }

    return out;
}

void ModifiedParamsResult::update_map_from_free_vector(
    std::unordered_map<std::string, std::vector<double>>& params_map,
    const std::vector<double>& free_vec
) const {
    size_t x_idx = 0;

    // Write values back by free-parameter order and original vector lengths.
    for (const auto& name : name_free) {
        const size_t param_len = structured.free.at(name).size();
        for (size_t k = 0; k < param_len; ++k) {
            params_map[name][k] = free_vec[x_idx++];
        }
    }
}

std::vector<int> ModifiedParamsResult::get_free_sizes() const {
    std::vector<int> sizes;
    sizes.reserve(name_free.size());

    // Return each free parameter vector length in name_free order.
    for (const auto& name : name_free) {
        sizes.push_back(static_cast<int>(structured.free.at(name).size()));
    }

    return sizes;
}
