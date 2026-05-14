#include "../include/modify_outputs.hpp"

#include <algorithm>

namespace modify_outputs {

/* ========================================================================== *
 *                         Base Name Ordering Utilities                        *
 * ========================================================================== */

// Build final base-name order:
// 1) keep user-specified order first;
// 2) append unseen parameter names from best_params in sorted order.
std::vector<std::string> ordered_base_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    // Start from user order to preserve explicit display preference.
    std::vector<std::string> out = user_order;

    // Track which names are already placed to avoid duplicates.
    std::unordered_map<std::string, bool> seen;
    for (const auto& key : out) {
        seen[key] = true;
    }

    // Collect names that exist in fitted params but were not listed by user.
    std::vector<std::string> remain;
    for (const auto& kv : best_params) {
        if (!seen.count(kv.first)) {
            remain.push_back(kv.first);
        }
    }

    // Sort remaining names to make output deterministic across hash orders.
    std::sort(remain.begin(), remain.end());

    // Append remaining names behind user-defined names.
    out.insert(out.end(), remain.begin(), remain.end());
    return out;
}

/* ========================================================================== *
 *                        Flat Name Expansion Utilities                        *
 * ========================================================================== */

// Expand ordered base names into flat column names:
// - scalar parameter:   "name"
// - vector parameter:   "name_1", "name_2", ...
std::vector<std::string> ordered_flat_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::size_t>& param_sizes,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    std::vector<std::string> out;

    // Reuse base ordering logic so both APIs stay consistent.
    std::vector<std::string> bases = ordered_base_names(
        user_order, best_params
    );

    // Expand each base name by its effective length.
    for (const auto& base : bases) {
        std::size_t p_len = 0;

        // Priority 1: declared size from parameter schema (if provided).
        auto its = param_sizes.find(base);
        if (its != param_sizes.end()) {
            p_len = its->second;
        }

        // Priority 2: actual fitted vector size may be larger; keep max.
        auto itb = best_params.find(base);
        if (itb != best_params.end()) {
            p_len = std::max(p_len, itb->second.size());
        }

        // Length 0/1 is treated as scalar to keep compact naming.
        if (p_len <= 1) {
            out.push_back(base);
        } else {
            // Multi-value parameter is expanded into 1-based indexed names.
            for (std::size_t j = 0; j < p_len; ++j) {
                out.push_back(base + "_" + std::to_string(j + 1));
            }
        }
    }
    return out;
}

} // namespace modify_outputs

