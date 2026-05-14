#include "../include/modify_outputs.hpp"

#include <algorithm>

namespace modify_outputs {

/* ========================================================================== *
 *                         Base Name Ordering Utilities                       *
 * ========================================================================== */

// 构建最终的基础参数名顺序:
// 1) 优先保留用户指定的顺序
// 2) 将 best_params 中未出现过的参数名按字母顺序追加到末尾
std::vector<std::string> ordered_base_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    // 从用户定义的顺序开始, 以保留显式的显示偏好
    std::vector<std::string> out = user_order;

    // 跟踪已放置的参数名, 以避免重复
    std::unordered_map<std::string, bool> seen;
    for (const auto& key : out) {
        seen[key] = true;
    }

    // 收集存在于拟合参数中但未被用户列出的名称.
    std::vector<std::string> remain;
    for (const auto& kv : best_params) {
        if (!seen.count(kv.first)) {
            remain.push_back(kv.first);
        }
    }

    // 对剩余的名称进行排序, 以确保在不同的哈希顺序下输出是确定性的
    std::sort(remain.begin(), remain.end());

    // 将剩余的参数名追加到用户定义的名称之后
    out.insert(out.end(), remain.begin(), remain.end());
    return out;
}

/* ========================================================================== *
 *                        Flat Name Expansion Utilities                       *
 * ========================================================================== */

// 将排序后的基础参数名展开为扁平的列名:
// - 标量参数:   "name"
// - 向量参数:   "name_1", "name_2", ...
std::vector<std::string> ordered_flat_names(
    const std::vector<std::string>& user_order,
    const std::unordered_map<std::string, std::size_t>& param_sizes,
    const std::unordered_map<std::string, std::vector<double>>& best_params
) {
    std::vector<std::string> out;

    // 复用基础排序逻辑, 以保持两个 API 的一致性
    std::vector<std::string> bases = ordered_base_names(
        user_order, best_params
    );

    // 根据其有效长度展开每个基础参数名
    for (const auto& base : bases) {
        std::size_t p_len = 0;

        // 优先级 1: 参数模式中声明的长度(如果提供了的话)
        auto its = param_sizes.find(base);
        if (its != param_sizes.end()) {
            p_len = its->second;
        }

        // 优先级 2: 实际拟合的向量长度可能更大; 取两者的最大值
        auto itb = best_params.find(base);
        if (itb != best_params.end()) {
            p_len = std::max(p_len, itb->second.size());
        }

        // 长度为 0 或 1 的参数被视为标量, 以保持紧凑的命名
        if (p_len <= 1) {
            out.push_back(base);
        } else {
            // 多值参数将被展开为基于 1 索引的名称
            for (std::size_t j = 0; j < p_len; ++j) {
                out.push_back(base + "_" + std::to_string(j + 1));
            }
        }
    }
    return out;
}

} // namespace modify_outputs
