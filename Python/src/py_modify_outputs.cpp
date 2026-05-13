#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace py_modify_outputs {

inline std::vector<std::string> ordered_base_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    std::vector<std::string> out = user_order;
    std::unordered_map<std::string, bool> seen;
    for (const auto& k : out) seen[k] = true;
    std::vector<std::string> remain;
    for (const auto& kv : best_params) {
        if (!seen.count(kv.first)) remain.push_back(kv.first);
    }
    std::sort(remain.begin(), remain.end());
    out.insert(out.end(), remain.begin(), remain.end());
    return out;
}

inline std::vector<std::string> ordered_flat_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, size_t>& param_sizes,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    std::vector<std::string> out;
    auto bases = ordered_base_names(user_order, best_params);
    for (const auto& b : bases) {
        size_t p_len = 0;
        auto its = param_sizes.find(b);
        if (its != param_sizes.end()) p_len = its->second;
        auto itb = best_params.find(b);
        if (itb != best_params.end()) p_len = std::max(p_len, itb->second.size());
        if (p_len <= 1) out.push_back(b);
        else for (size_t j = 0; j < p_len; ++j) out.push_back(b + "_" + std::to_string(j + 1));
    }
    return out;
}

} // namespace py_modify_outputs

