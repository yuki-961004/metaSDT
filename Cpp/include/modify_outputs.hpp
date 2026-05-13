#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace modify_outputs {

std::vector<std::string> ordered_base_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::vector<double>>& best_params
);

std::vector<std::string> ordered_flat_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::size_t>& param_sizes,
    const std::unordered_map<std::string, std::vector<double>>& best_params
);

} // namespace modify_outputs

