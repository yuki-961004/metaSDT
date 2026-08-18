#include <metaSDT/info_data.hpp>
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
    std::vector<std::pair<std::string, std::string>> targets = {
        {"subid", "^(.*_)?(subj|subject|subid|participant|sub|id)$"},
        {"trial", "^(.*_)?trial.*"},
        {"block", "^(.*_)?block.*"},
        {"stim", ".*(stim|target|signal).*"},
        {"intn", ".*(intensity|strength|coherence|snr|contrast).*"},
        {"resp", ".*(resp|decision|choice).*"},
        {"conf", ".*(conf|rating).*"}
    };

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

    for (const auto& tgt : targets) {
        const std::string& role = tgt.first;
        const std::string& pattern_str = tgt.second;

        if (!result.colnames.count(role)) {
            std::regex re(pattern_str, std::regex_constants::icase);
            for (const auto& kv : df) {
                if (std::regex_match(kv.first, re)) {
                    result.colnames[role] = kv.first;
                    break;
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
            missing_roles.push_back(role);
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

    const bool has_subid_col = result.colnames.count("subid");
    const std::string subid_col = has_subid_col ? result.colnames.at("subid") : "";

    for (size_t i = 0; i < n_rows; ++i) {
        double subid = has_subid_col ? df.at(subid_col)[i] : 1.0;
        
        DataInfoSubject& subj = result.subjects[subid];
        subj.raw.push_back(i);
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
            
            subj.condition[cond_str].push_back(i);
        }

        if (!diff_col.empty() && df.count(diff_col)) {
            double d_val = df.at(diff_col)[i];
            std::ostringstream oss;
            oss << d_val;
            std::string diff_str = oss.str();
            
            subj.difficulty[diff_str].push_back(i);
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
