#include "../include/info_data.hpp"
#include <regex>
#include <stdexcept>
#include <sstream>
#include <unordered_set>

DataInfoResult info_data(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames
) {
    DataInfoResult result;

/* ========================================================================== *
 *                          1. Column Role Matching                           *
 * ========================================================================== */
    // 使用简写作为标准角色名.
    std::vector<std::pair<std::string, std::string>> targets = {
        {"subid", ".*(sub|id|participant).*"},
        {"trial", ".*trial.*"},
        {"block", ".*block.*"},
        {"stim", ".*(stim|target|signal).*"},
        {"intn", ".*(intensity|strength|coherence|snr|contrast).*"},
        {"resp", ".*(resp|decision|choice).*"},
        {"conf", ".*(conf|rating).*"}
    };

    // 支持用户传入更为口语化的键名 (aliases)
    auto user_colnames = colnames;

    if (user_colnames.count("subject") && !user_colnames.count("subid")) {
        user_colnames["subid"] = user_colnames.at("subject");
    }
    if (user_colnames.count("condition") && !user_colnames.count("cond")) {
        user_colnames["cond"] = user_colnames.at("condition");
    }
    if (user_colnames.count("difficulty") && !user_colnames.count("diff")) {
        user_colnames["diff"] = user_colnames.at("difficulty");
    }

    // 将用户显式指定的映射优先放入结果
    for (const auto& kv : user_colnames) {
        result.colnames[kv.first] = kv.second;
    }

    for (const auto& tgt : targets) {
        const std::string& role = tgt.first;
        const std::string& pattern_str = tgt.second;

        // 如果该角色还没有被用户显式指定，则尝试正则匹配.
        if (!result.colnames.count(role)) {
            std::regex re(pattern_str, std::regex_constants::icase);
            for (const auto& kv : df) {
                if (std::regex_match(kv.first, re)) {
                    result.colnames[role] = kv.first;
                    break; // 找到第一个匹配的就退出当前角色的寻找
                }
            }
        }
    }

/* ========================================================================== *
 *                        2. Required-Role Validation                         *
 * ========================================================================== */
    std::vector<std::string> missing_roles;
    for (const auto& tgt : targets) {
        const std::string& role = tgt.first;
        if (!result.colnames.count(role)) {
            if (role == "subid") {
                throw std::invalid_argument(
                    "Error: Could not identify the critical column for '" + 
                    role + "'. Please specify it in `colnames`."
                );
            } else {
                missing_roles.push_back(role);
            }
        }
    }

    if (!missing_roles.empty()) {
        std::string missing_str = "";
        for (size_t i = 0; i < missing_roles.size(); ++i) {
            missing_str += "'" + missing_roles[i] + "'";
            if (i < missing_roles.size() - 1) missing_str += ", ";
        }
        result.messages.push_back(
            "Note: Could not identify column(s) for " + missing_str + 
            ". These roles will be ignored."
        );
    }

    // 校验所有映射的列名是否在数据集中真实存在.
    for (const auto& kv : result.colnames) {
        if (!df.count(kv.second)) {
            throw std::invalid_argument(
                "Error: The assigned column '" + kv.second + 
                "' for role '" + kv.first + "' does not exist in the dataset."
            );
        }
    }

    std::string subid_col = result.colnames.at("subid");

    std::string block_col = "";
    if (result.colnames.count("block")) {
        block_col = result.colnames.at("block");
    }
    
    std::string cond_col = result.colnames.count("cond") ? 
                           result.colnames.at("cond") : "";
    std::string diff_col = result.colnames.count("diff") ? 
                           result.colnames.at("diff") : "";

/* ========================================================================== *
 *                       3. Subject View Construction                         *
 * ========================================================================== */
    // 获取总行数(基于 subid 列的长度)
    size_t n_rows = df.at(subid_col).size();

    std::unordered_map<double, std::unordered_set<int>> subject_blocks;

    // 遍历原始数据的每一行.
    for (size_t i = 0; i < n_rows; ++i) {
        double subid = df.at(subid_col)[i];
        
        // 提取该被试对应的引用，如果这是他/她的第一条数据，.
        // 将自动在 map 中初始化.
        DataInfoSubject& subj = result.subjects[subid];

        // 添加极其省内存的行号索引.
        subj.raw.push_back(i);
        subj.info.n_trials++;

        // 处理 block 统计信息.
        if (!block_col.empty() && df.count(block_col)) {
            int b = static_cast<int>(df.at(block_col)[i]);
            subject_blocks[subid].insert(b); // 记录出现过的 block 编号
        }

        // 处理 condition 分组.
        if (!cond_col.empty() && df.count(cond_col)) {
            std::ostringstream cond_key;
            cond_key << df.at(cond_col)[i];
            subj.condition[cond_key.str()].push_back(i);
        }
        
        // 处理 difficulty 分组.
        if (!diff_col.empty() && df.count(diff_col)) {
            std::ostringstream diff_key;
            diff_key << df.at(diff_col)[i];
            subj.difficulty[diff_key.str()].push_back(i);
        }
    }

/* ========================================================================== *
 *                           4. Final Aggregation                             *
 * ========================================================================== */
    for (auto& kv : result.subjects) {
        kv.second.info.n_blocks = subject_blocks[kv.first].size();
    }

    return result;
}
