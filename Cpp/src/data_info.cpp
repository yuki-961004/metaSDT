#include "../include/data_info.hpp"
#include <regex>
#include <stdexcept>
#include <sstream>
#include <unordered_set>

DataInfoResult data_info_core(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const std::vector<std::string>& condition
) {
    DataInfoResult result;

    // ==========================================================
    // 1. 智能列名匹配 (正则表达式字典)
    // ==========================================================
    // 使用简写作为标准角色名
    std::vector<std::pair<std::string, std::string>> targets = {
        {"subid", ".*(sub|id|participant).*"},
        {"trial", ".*trial.*"},
        {"block", ".*block.*"},
        {"stim", ".*(stim|target|signal).*"},
        {"intn", ".*(intensity|strength|coherence|snr|contrast).*"},
        {"resp", ".*(resp|decision|choice).*"},
        {"conf", ".*(conf|rating).*"}
    };

    for (const auto& tgt : targets) {
        const std::string& role = tgt.first;
        const std::string& pattern_str = tgt.second;

        // 如果用户在 colnames 中明确指派了该列名，直接采用
        if (colnames.count(role)) {
            result.colnames[role] = colnames.at(role);
        } else {
            // 否则遍历数据框的所有列名，进行正则匹配忽略大小写寻找
            std::regex re(pattern_str, std::regex_constants::icase);
            for (const auto& kv : df) {
                if (std::regex_match(kv.first, re)) {
                    result.colnames[role] = kv.first;
                    break; // 找到第一个匹配的就退出当前角色的寻找
                }
            }
        }
    }

    // ==========================================================
    // 2. 校验关键列与缺失处理
    // ==========================================================
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
        result.warnings.push_back(
            "Warning: Could not identify column(s) for " + missing_str + 
            ". These roles will be ignored."
        );
    }

    std::string subid_col = result.colnames.at("subid");
    if (!df.count(subid_col)) {
        throw std::invalid_argument(
            "Error: The assigned 'subid' column '" + subid_col + 
            "' does not exist in the dataset."
        );
    }

    std::string block_col = "";
    if (result.colnames.count("block")) {
        block_col = result.colnames.at("block");
    }

    // ==========================================================
    // 3. 按被试拆分数据集并生成轻量级视图 (View)
    // ==========================================================
    // 获取总行数 (基于 subid 列的长度)
    size_t n_rows = df.at(subid_col).size();

    std::unordered_map<double, std::unordered_set<int>> subject_blocks;

    // 遍历原始数据的每一行
    for (size_t i = 0; i < n_rows; ++i) {
        double subid = df.at(subid_col)[i];
        
        // 提取该被试对应的引用，如果这是他/她的第一条数据，
        // 将自动在 map 中初始化
        DataInfoSubject& subj = result.subjects[subid];

        // 添加极其省内存的行号索引
        subj.raw.push_back(i);
        subj.info.n_trials++;

        // 处理 block 统计信息
        if (!block_col.empty() && df.count(block_col)) {
            int b = static_cast<int>(df.at(block_col)[i]);
            subject_blocks[subid].insert(b); // 记录出现过的 block 编号
        }

        // 处理条件分组 (Condition Grouping)
        // 将所有条件的数值拼接成一个字符串键 
        // (例如 condition = [stim, block], 则键为 "1_2")
        if (!condition.empty()) {
            std::ostringstream cond_key;
            for (size_t c = 0; c < condition.size(); ++c) {
                const std::string& c_col = condition[c];
                if (df.count(c_col)) {
                    cond_key << df.at(c_col)[i];
                } else {
                    cond_key << "NA";
                }
                if (c < condition.size() - 1) cond_key << "_";
            }
            // 在对应条件键下，放入该行号
            subj.condition[cond_key.str()].push_back(i);
        }
    }

    // ==========================================================
    // 4. 汇总与最终统计计算
    // ==========================================================
    for (auto& kv : result.subjects) {
        kv.second.info.n_blocks = subject_blocks[kv.first].size();
    }

    return result;
}