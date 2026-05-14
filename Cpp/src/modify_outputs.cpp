#include "../include/modify_outputs.hpp"

#include <algorithm>

namespace modify_outputs {

std::vector<std::string> ordered_base_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    std::vector<std::string> out = user_order;
    std::unordered_map<std::string, bool> seen;
    for (const auto& key : out) {
        seen[key] = true;
    }

    std::vector<std::string> remain;
    for (const auto& kv : best_params) {
        if (!seen.count(kv.first)) {
            remain.push_back(kv.first);
        }
    }
    std::sort(remain.begin(), remain.end());
    out.insert(out.end(), remain.begin(), remain.end());
    return out;
}

std::vector<std::string> ordered_flat_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::size_t>& param_sizes,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    std::vector<std::string> out;
    std::vector<std::string> bases = ordered_base_names(
        user_order, best_params
    );

    for (const auto& base : bases) {
        std::size_t p_len = 0;
        auto its = param_sizes.find(base);
        if (its != param_sizes.end()) {
            p_len = its->second;
        }
        auto itb = best_params.find(base);
        if (itb != best_params.end()) {
            p_len = std::max(p_len, itb->second.size());
        }

        if (p_len <= 1) {
            out.push_back(base);
        } else {
            for (std::size_t j = 0; j < p_len; ++j) {
                out.push_back(base + "_" + std::to_string(j + 1));
            }
        }
    }
    return out;
}

} // namespace modify_outputs

