#include "../include/matrix_freq.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <cmath>

namespace {

/* ========================================================================== *
 *                              Internal Helpers                              *
 * ========================================================================== */

// 在向量中查找目标值的索引，若未找到则返回 -1.
int find_index(const std::vector<double>& vec, double target) {
    auto it = std::find(vec.begin(), vec.end(), target);
    if (it != vec.end()) {
        return static_cast<int>(std::distance(vec.begin(), it));
    }
    return -1;
}

// 获取向量中去重并排序后的唯一值集合.
std::vector<double> get_unique_sorted(const std::vector<double>& vec) {
    std::set<double> s(vec.begin(), vec.end());
    return std::vector<double>(s.begin(), s.end());
}

// 将浮点数格式化为字符串，并移除尾随的零和小数点.
std::string format_double(double val) {
    std::string s = std::to_string(val);
    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    if (!s.empty() && s.back() == '.') {
        s.pop_back();
    }
    return s;
}

/* ========================================================================== *
 *                         Data Matrix Preparation                            *
 * ========================================================================== */

// 将离散的一维数据列组合成一个规整的二维数值矩阵.
// 列顺序固定为：[stim, resp, conf, diff].
std::vector<std::vector<double>> prep_num_mat(
    const std::vector<double>& stim,
    const std::vector<double>& resp,
    const std::vector<double>* conf,
    const std::vector<double>* diff
) {
    // 获取总行数，并校验 stim 与 resp 长度是否一致.
    size_t n_rows = stim.size();
    if (resp.size() != n_rows) {
        throw std::invalid_argument(
            "Error: 'stim' and 'resp' must have the same length."
        );
    }

    // 初始化输出矩阵，默认用 0.0 填充.
    std::vector<std::vector<double>> out_mat(
        n_rows, std::vector<double>(4, 0.0)
    );

    // 校验可选的 conf 与 diff 列长度.
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

    // 逐行填充数据；若 conf 或 diff 为空，则默认使用 1.0 占位.
    for (size_t i = 0; i < n_rows; ++i) {
        out_mat[i][0] = stim[i];
        out_mat[i][1] = resp[i];
        out_mat[i][2] = (conf != nullptr) ? (*conf)[i] : 1.0;
        out_mat[i][3] = (diff != nullptr) ? (*diff)[i] : 1.0;
    }

    return out_mat;
}

/* ========================================================================== *
 *                       Confidence Rating Processing                         *
 * ========================================================================== */

// 处理置信度评分，将其映射到模型期望的离散箱 (bins) 中.
void process_confidence_ratings(
    const std::vector<double>* conf,
    const std::unordered_map<std::string, std::vector<double>>* std_params,
    std::vector<double>& processed_conf,
    const std::vector<double>*& final_conf_ptr
) {
    // 如果未提供 conf 或相关参数，则跳过处理.
    if (conf == nullptr ||
        std_params == nullptr ||
        !std_params->count("n_conf") ||
        std_params->at("n_conf").empty()) {
        return;
    }

    // 计算期望的响应类别总数与每种响应的置信度箱数.
    int n_criteria = static_cast<int>(std_params->at("n_conf")[0]);
    int num_bins = (n_criteria + 1) / 2;
    if (num_bins <= 1) {
        return;
    }

    // 获取当前数据中唯一的置信度值.
    auto unique_conf = get_unique_sorted(*conf);
    if (unique_conf.size() <= static_cast<size_t>(num_bins)) {
        return;
    }

    // 检查数据是否为离散的整数评分，且跨度在合理范围内 (<= 25).
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

    // 如果数据是典型的离散评分，但与模型配置的箱数不符，抛出错误提示.
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

    // 对于连续的置信度评分，将其线性映射并离散化为期望的箱数.
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

/* ========================================================================== *
 *                      Main Frequency Matrix Builder                         *
 * ========================================================================== */

// 构建用于似然计算的三维观测频数矩阵 (diff x stim x resp_conf).
MatrixFreq matrix_freq(
    const std::vector<double>& stim,
    const std::vector<double>& resp,
    const std::vector<double>* conf,
    const std::vector<double>* diff,
    const std::unordered_map<std::string, std::vector<double>>* std_params
) {
    // 预处理置信度列，应用归一化和离散化映射.
    std::vector<double> processed_conf;
    const std::vector<double>* final_conf_ptr = conf;
    process_confidence_ratings(
        /*conf=*/conf,
        /*std_params=*/std_params,
        /*processed_conf=*/processed_conf,
        /*final_conf_ptr=*/final_conf_ptr
    );

    // 构建行对齐的内部数据矩阵.
    auto num_mat = prep_num_mat(stim, resp, final_conf_ptr, diff);
    size_t n_rows = num_mat.size();

    // 拆解各列以便于独立提取唯一值.
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

    // 获取各维度的唯一取值集合.
    auto unique_stim = get_unique_sorted(stim_col);
    auto unique_resp = get_unique_sorted(resp_col);
    auto unique_conf = get_unique_sorted(conf_col);
    auto unique_diff = get_unique_sorted(diff_col);

    // 根据参数配置，覆盖并严格设定期望的置信度箱数.
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

    // 计算各维度的大小，以及展平后的响应-置信度列数.
    size_t n_stim = unique_stim.size();
    size_t n_resp = unique_resp.size();
    size_t n_conf = unique_conf.size();
    size_t n_diffs = unique_diff.size();
    size_t n_cols_out = n_resp * n_conf;

    // 初始化全零的三维频数矩阵.
    MatrixFreq result;
    result.freq_mat.assign(
        n_diffs,
        std::vector<std::vector<double>>(
            n_stim,
            std::vector<double>(n_cols_out, 0.0)
        )
    );

    // 遍历所有数据行，将观测结果累加到频数矩阵对应的单元格中.
    for (size_t i = 0; i < n_rows; ++i) {
        // 定位当前观测所在的维度索引.
        int row_idx = find_index(unique_stim, num_mat[i][0]);
        int resp_idx = find_index(unique_resp, num_mat[i][1]);
        int conf_idx = find_index(unique_conf, num_mat[i][2]);
        
        // 对于未精确匹配的置信度，尝试做就近的整数映射.
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

        // 根据响应类型决定列索引排布，resp=0 时置信度逆序，resp=1 时正序.
        int col_idx;
        if (resp_idx == 0) {
            col_idx = static_cast<int>((n_conf - 1) - conf_idx);
        } else {
            col_idx = static_cast<int>(resp_idx * n_conf + conf_idx);
        }
        result.freq_mat[diff_idx][row_idx][col_idx] += 1.0;
    }

    // 生成具有可读性的维度、行、列的元数据标签.
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
