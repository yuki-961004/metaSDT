#include <string>
#include <unordered_map>
#include <vector>
#include "../../Cpp/include/modify_outputs.hpp"

namespace py_modify_outputs {

inline std::vector<std::string> ordered_base_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    return modify_outputs::ordered_base_names(user_order, best_params);
}

inline std::vector<std::string> ordered_flat_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, size_t>& param_sizes,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    return modify_outputs::ordered_flat_names(
        user_order, param_sizes, best_params
    );
}

} // namespace py_modify_outputs
