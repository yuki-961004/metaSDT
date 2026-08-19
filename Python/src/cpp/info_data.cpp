#include <metaSDT/info_data.hpp>

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

// 将字符串转为小写并去除首尾空白字符
std::string normalize_name(const std::string& input) {
    std::string s = input;
    // 转换为小写
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    // 去除前导空白
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    // 去除尾随空白
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
    return s;
}

// 检查归一化名称是否精确属于给定的候选集合
bool matches_any(
    const std::string& name,
    const std::initializer_list<std::string>& candidates
) {
    for (const auto& cand : candidates) {
        if (name == cand) {
            return true;
        }
    }
    return false;
}

// 获取某个归一化列名所能匹配的所有可能标准角色
std::vector<std::string> identify_roles_for_column(const std::string& raw_name) {
    const std::string norm = normalize_name(raw_name);
    std::vector<std::string> matched_roles;

    // 1. 被试列 (subid): 仅严格匹配完整的被试标识符名称，绝不使用模糊子串
    if (matches_any(norm, {"subid", "subject", "subject_id", "sub", "subj", "subj_id", "participant"})) {
        matched_roles.push_back("subid");
    }

    // 2. 试次列 (trial)
    if (matches_any(norm, {"trial", "trial_id", "trial_num", "trial_no", "trials", "trial_idx"})) {
        matched_roles.push_back("trial");
    }

    // 3. Block 列 (block)
    if (matches_any(norm, {"block", "block_id", "block_num", "block_no", "blocks", "block_idx"})) {
        matched_roles.push_back("block");
    }

    // 4. 刺激列 (stim)
    if (matches_any(norm, {"stim", "stimulus", "target", "signal", "stim_type"})) {
        matched_roles.push_back("stim");
    }

    // 5. 强度列 (intn)
    if (matches_any(norm, {"intn", "intensity", "strength", "coherence", "snr", "contrast"})) {
        matched_roles.push_back("intn");
    }

    // 6. 反应列 (resp)
    if (matches_any(norm, {"resp", "response", "decision", "choice"})) {
        matched_roles.push_back("resp");
    }

    // 7. 置信度列 (conf)
    if (matches_any(norm, {"conf", "confidence", "rating", "conf_rating"})) {
        matched_roles.push_back("conf");
    }

    // 8. 难度列 (diff)
    if (matches_any(norm, {"diff", "difficulty", "diff_level"})) {
        matched_roles.push_back("diff");
    }

    // 9. 条件列 (cond)
    if (matches_any(norm, {"cond", "condition"})) {
        matched_roles.push_back("cond");
    }

    // 10. 证据列 (evidence)
    if (matches_any(norm, {"evidence", "ev"})) {
        matched_roles.push_back("evidence");
    }

    return matched_roles;
}

} // namespace

DataInfoResult info_data(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames
) {
    DataInfoResult result;

/* ========================================================================== *
 *                          1. Column Role Matching                           *
 * ========================================================================== */
    // 首先引入用户明确指定的列名别名映射
    std::unordered_map<std::string, std::string> user_colnames = colnames;

    if (user_colnames.count("subject") && !user_colnames.count("subid")) {
        user_colnames["subid"] = user_colnames.at("subject");
    }
    if (user_colnames.count("condition") && !user_colnames.count("cond")) {
        user_colnames["cond"] = user_colnames.at("condition");
    }
    if (user_colnames.count("difficulty") && !user_colnames.count("diff")) {
        user_colnames["diff"] = user_colnames.at("difficulty");
    }

    for (const auto& kv : user_colnames) {
        result.colnames[kv.first] = kv.second;
    }

    // 针对用户未明确指定的角色，进行严格且无歧义的自动解析
    std::unordered_map<std::string, std::vector<std::string>> role_to_candidates;

    for (const auto& kv : df) {
        const std::string& col_name = kv.first;
        const std::vector<std::string> roles = identify_roles_for_column(col_name);

        if (roles.size() > 1) {
            // 检查用户是否已经显式指定了冲突角色的映射
            int unresolved_count = 0;
            for (const auto& r : roles) {
                if (!result.colnames.count(r) || result.colnames[r] == col_name) {
                    unresolved_count++;
                }
            }
            if (unresolved_count > 1) {
                throw std::invalid_argument(
                    "Error: Ambiguous column name '" + col_name +
                    "' matches multiple roles. Please explicitly specify "
                    "`colnames`."
                );
            }
        }

        for (const auto& role : roles) {
            if (!result.colnames.count(role)) {
                role_to_candidates[role].push_back(col_name);
            }
        }
    }

    for (const auto& kv : role_to_candidates) {
        const std::string& role = kv.first;
        const std::vector<std::string>& candidates = kv.second;

        if (candidates.size() > 1) {
            throw std::invalid_argument(
                "Error: Multiple dataset columns matched role '" + role +
                "'. Please specify the desired column in `colnames`."
            );
        } else if (candidates.size() == 1) {
            result.colnames[role] = candidates[0];
        }
    }

/* ========================================================================== *
 *                        2. Required-Role Validation                         *
 * ========================================================================== */
    for (const auto& kv : result.colnames) {
        if (!df.count(kv.second)) {
            throw std::invalid_argument(
                "Error: The assigned column '" + kv.second + 
                "' for role '" + kv.first + "' does not exist in the dataset."
            );
        }
    }

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
    size_t n_rows = 0;
    if (!df.empty()) {
        n_rows = df.begin()->second.size();
    }

    std::unordered_map<double, std::unordered_set<int>> subject_blocks;

    // 当且仅当存在明确的 subid 列时才按被试切分；否则退化为唯一的合成单被试 (1.0)
    const bool has_subid_col = result.colnames.count("subid");
    const std::string subid_col = has_subid_col ? result.colnames.at("subid") : "";

    for (size_t i = 0; i < n_rows; ++i) {
        double subid = has_subid_col ? df.at(subid_col)[i] : 1.0;
        
        DataInfoSubject& subj = result.subjects[subid];
        subj.raw.push_back(static_cast<int>(i));
        subj.info.n_trials++;

        if (!block_col.empty() && df.count(block_col)) {
            int b = static_cast<int>(df.at(block_col)[i]);
            subject_blocks[subid].insert(b);
        }

        if (!cond_col.empty() && df.count(cond_col)) {
            double c_val = df.at(cond_col)[i];
            std::ostringstream oss;
            oss << c_val;
            std::string cond_str = oss.str();
            
            subj.condition[cond_str].push_back(static_cast<int>(i));
        }

        if (!diff_col.empty() && df.count(diff_col)) {
            double d_val = df.at(diff_col)[i];
            std::ostringstream oss;
            oss << d_val;
            std::string diff_str = oss.str();
            
            subj.difficulty[diff_str].push_back(static_cast<int>(i));
        }
    }

    for (auto& kv : result.subjects) {
        double subid = kv.first;
        DataInfoSubject& subj = kv.second;
        
        if (!block_col.empty()) {
            subj.info.n_blocks = static_cast<int>(subject_blocks[subid].size());
        } else {
            subj.info.n_blocks = 1;
        }
    }

    return result;
}
