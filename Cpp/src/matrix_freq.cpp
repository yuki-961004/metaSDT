#include "../include/matrix_freq.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <cmath>

namespace {

int find_index(const std::vector<double>& vec, double target) {
    auto it = std::find(vec.begin(), vec.end(), target);
    if (it != vec.end()) {
        return static_cast<int>(std::distance(vec.begin(), it));
    }
    return -1;
}

std::vector<double> get_unique_sorted(const std::vector<double>& vec) {
    std::set<double> s(vec.begin(), vec.end());
    return std::vector<double>(s.begin(), s.end());
}

std::string format_double(double val) {
    std::string s = std::to_string(val);
    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    if (!s.empty() && s.back() == '.') {
        s.pop_back();
    }
    return s;
}

std::vector<std::vector<double>> prep_num_mat(
    const std::vector<double>& stim,
    const std::vector<double>& resp,
    const std::vector<double>* conf,
    const std::vector<double>* diff
) {
    size_t n_rows = stim.size();
    if (resp.size() != n_rows) {
        throw std::invalid_argument(
            "Error: 'stim' and 'resp' must have the same length."
        );
    }

    std::vector<std::vector<double>> out_mat(
        n_rows, std::vector<double>(4, 0.0)
    );

    if (conf != nullptr && conf->size() != n_rows) {
        throw std::invalid_argument(
            "Error: 'conf' must have the same length as 'stim' and 'resp'."
        );
    }
    if (diff != nullptr && diff->size() != n_rows) {
        throw std::invalid_argument(
            "Error: 'diff' must have the same length."
        );
    }

    for (size_t i = 0; i < n_rows; ++i) {
        out_mat[i][0] = stim[i];
        out_mat[i][1] = resp[i];
        out_mat[i][2] = (conf != nullptr) ? (*conf)[i] : 1.0;
        out_mat[i][3] = (diff != nullptr) ? (*diff)[i] : 1.0;
    }

    return out_mat;
}

void process_confidence_ratings(
    const std::vector<double>* conf,
    const std::unordered_map<std::string, std::vector<double>>* std_params,
    std::vector<double>& processed_conf,
    const std::vector<double>*& final_conf_ptr
) {
    if (conf == nullptr ||
        std_params == nullptr ||
        !std_params->count("n_conf") ||
        std_params->at("n_conf").empty()) {
        return;
    }

    int n_criteria = static_cast<int>(std_params->at("n_conf")[0]);
    int num_bins = (n_criteria + 1) / 2;
    if (num_bins <= 1) {
        return;
    }

    auto unique_conf = get_unique_sorted(*conf);
    if (unique_conf.size() <= static_cast<size_t>(num_bins)) {
        return;
    }

    double min_c = unique_conf.front();
    double max_c = unique_conf.back();
    bool is_integer = true;
    for (double v : unique_conf) {
        if (std::abs(v - std::round(v)) > 1e-6) {
            is_integer = false;
            break;
        }
    }
    double span = max_c - min_c + 1.0;
    bool is_discrete = is_integer && (span <= 25.0);

    if (is_discrete) {
        int expected_c_conf = static_cast<int>(unique_conf.size()) - 1;
        throw std::invalid_argument(
            "Error: Mismatch between model parameters and discrete "
            "confidence data.\nYour 'conf' data contains " +
            std::to_string(unique_conf.size()) + " discrete levels (from " +
            format_double(min_c) + " to " + format_double(max_c) + "), "
            "but the model is configured to expect only " +
            std::to_string(num_bins) + " levels per response.\n"
            "If you intend to fit " +
            std::to_string(unique_conf.size()) +
            " confidence levels, you must provide exactly " +
            std::to_string(expected_c_conf) + " 'c_conf' boundaries.\n"
            "(Note: If your raw data is a 1-N rating scale, ensure it is "
            "split into a binary 'resp' column and a magnitude 'conf' "
            "column first)."
        );
    }

    if (max_c > min_c) {
        processed_conf.reserve(conf->size());
        for (double c : *conf) {
            double norm_c = (c - min_c) / (max_c - min_c);
            int bin = static_cast<int>(std::floor(norm_c * num_bins)) + 1;
            if (bin > num_bins) {
                bin = num_bins;
            }
            processed_conf.push_back(static_cast<double>(bin));
        }
        final_conf_ptr = &processed_conf;
    }
}

} // namespace

MatrixFreq matrix_freq(
    const std::vector<double>& stim,
    const std::vector<double>& resp,
    const std::vector<double>* conf,
    const std::vector<double>* diff,
    const std::unordered_map<std::string, std::vector<double>>* std_params
) {
    std::vector<double> processed_conf;
    const std::vector<double>* final_conf_ptr = conf;
    process_confidence_ratings(
        /*conf=*/conf,
        /*std_params=*/std_params,
        /*processed_conf=*/processed_conf,
        /*final_conf_ptr=*/final_conf_ptr
    );

    auto num_mat = prep_num_mat(stim, resp, final_conf_ptr, diff);
    size_t n_rows = num_mat.size();

    std::vector<double> stim_col;
    std::vector<double> resp_col;
    std::vector<double> conf_col;
    std::vector<double> diff_col;
    for (size_t i = 0; i < n_rows; ++i) {
        stim_col.push_back(num_mat[i][0]);
        resp_col.push_back(num_mat[i][1]);
        conf_col.push_back(num_mat[i][2]);
        diff_col.push_back(num_mat[i][3]);
    }

    auto unique_stim = get_unique_sorted(stim_col);
    auto unique_resp = get_unique_sorted(resp_col);
    auto unique_conf = get_unique_sorted(conf_col);
    auto unique_diff = get_unique_sorted(diff_col);

    int expected_conf_bins = -1;
    if (std_params != nullptr &&
        std_params->count("n_conf") &&
        !std_params->at("n_conf").empty()) {
        int n_criteria = static_cast<int>(std_params->at("n_conf")[0]);
        expected_conf_bins = (n_criteria + 1) / 2;
        if (expected_conf_bins > 1) {
            unique_conf.clear();
            unique_conf.reserve(static_cast<size_t>(expected_conf_bins));
            for (int lv = 1; lv <= expected_conf_bins; ++lv) {
                unique_conf.push_back(static_cast<double>(lv));
            }
        }
    }

    size_t n_stim = unique_stim.size();
    size_t n_resp = unique_resp.size();
    size_t n_conf = unique_conf.size();
    size_t n_diffs = unique_diff.size();
    size_t n_cols_out = n_resp * n_conf;

    MatrixFreq result;
    result.freq_mat.assign(
        n_diffs,
        std::vector<std::vector<double>>(
            n_stim,
            std::vector<double>(n_cols_out, 0.0)
        )
    );

    for (size_t i = 0; i < n_rows; ++i) {
        int row_idx = find_index(unique_stim, num_mat[i][0]);
        int resp_idx = find_index(unique_resp, num_mat[i][1]);
        int conf_idx = find_index(unique_conf, num_mat[i][2]);
        if (conf_idx < 0 && expected_conf_bins > 1) {
            int mapped = static_cast<int>(std::round(num_mat[i][2]));
            if (mapped < 1) {
                mapped = 1;
            }
            if (mapped > expected_conf_bins) {
                mapped = expected_conf_bins;
            }
            conf_idx = mapped - 1;
        }
        int diff_idx = find_index(unique_diff, num_mat[i][3]);

        int col_idx;
        if (resp_idx == 0) {
            col_idx = static_cast<int>((n_conf - 1) - conf_idx);
        } else {
            col_idx = static_cast<int>(resp_idx * n_conf + conf_idx);
        }
        result.freq_mat[diff_idx][row_idx][col_idx] += 1.0;
    }

    for (double diff_val : unique_diff) {
        result.dim_names.push_back("diff_" + format_double(diff_val));
    }
    for (double stim_val : unique_stim) {
        result.row_names.push_back("stim_" + format_double(stim_val));
    }
    for (size_t r = 0; r < n_resp; ++r) {
        for (size_t c_loop = 0; c_loop < n_conf; ++c_loop) {
            size_t c = (r == 0) ? (n_conf - 1 - c_loop) : c_loop;
            std::string r_str = format_double(unique_resp[r]);
            if (conf == nullptr) {
                result.col_names.push_back("resp_" + r_str);
            } else {
                std::string c_str = format_double(unique_conf[c]);
                result.col_names.push_back(
                    "resp_" + r_str + "_conf_" + c_str
                );
            }
        }
    }

    return result;
}
