#include "../include/matrix_freq.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace matrix_freq_helper {

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
        // 去除尾随的 0
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        // 如果最后是小数点，也一并去除 (例如 "1." 变成 "1")
        if (!s.empty() && s.back() == '.') {
            s.pop_back();
        }
        return s;
    }

    std::vector<std::vector<double>> prep_num_mat(
        const std::vector<double>& stim,
        const std::vector<double>& resp,
        const std::vector<double>* conf) {

        size_t n_rows = stim.size();
        if (resp.size() != n_rows) {
            throw std::invalid_argument(
                "Error: 'stim' and 'resp' must have the same length."
            );
        }

        // 预分配 N 行 3 列的矩阵
        std::vector<std::vector<double>> out_mat(
            n_rows, std::vector<double>(3, 0.0)
        );

        if (conf != nullptr) {
            if (conf->size() != n_rows) {
                throw std::invalid_argument(
                    "Error: 'conf' must have the same length as "
                    "'stim' and 'resp'."
                );
            }
        }

        for (size_t i = 0; i < n_rows; ++i) {
            out_mat[i][0] = stim[i];
            out_mat[i][1] = resp[i];
            if (conf != nullptr) {
                out_mat[i][2] = (*conf)[i];
            } else {
                // 退化为经典的信号检测论频数矩阵
                out_mat[i][2] = 1.0; 
            }
        }
        return out_mat;
    }
}

MatrixFreq matrix_freq(
    const std::vector<double>& stim,
    const std::vector<double>& resp,
    const std::vector<double>* conf) {

    auto num_mat = matrix_freq_helper::prep_num_mat(
        stim, resp, conf
    );
    size_t n_rows = num_mat.size();

    // 分离列以便于获取 unique 值
    std::vector<double> sig_col, dec_col, conf_col;
    for (size_t i = 0; i < n_rows; ++i) {
        sig_col.push_back(num_mat[i][0]);
        dec_col.push_back(num_mat[i][1]);
        conf_col.push_back(num_mat[i][2]);
    }

    auto unique_sig = matrix_freq_helper::get_unique_sorted(sig_col);
    auto unique_dec = matrix_freq_helper::get_unique_sorted(dec_col);
    auto unique_conf = matrix_freq_helper::get_unique_sorted(conf_col);

    size_t n_sig = unique_sig.size();
    size_t n_dec = unique_dec.size();
    size_t n_conf = unique_conf.size();
    size_t n_cols_out = n_dec * n_conf;

    // 预分配频数矩阵
    MatrixFreq result;
    result.freq_mat.assign(
        n_sig, std::vector<double>(n_cols_out, 0.0)
    );

    // 遍历累加频数
    for (size_t i = 0; i < n_rows; ++i) {
        int row_idx = matrix_freq_helper::find_index(
            unique_sig, num_mat[i][0]
        );
        int dec_idx = matrix_freq_helper::find_index(
            unique_dec, num_mat[i][1]
        );
        int conf_idx = matrix_freq_helper::find_index(
            unique_conf, num_mat[i][2]
        );

        // C++ 是 0 索引，平铺映射公式: dec_idx * n_conf + conf_idx
        int col_idx = dec_idx * n_conf + conf_idx;
        result.freq_mat[row_idx][col_idx] += 1.0;
    }

    // 添加行名和列名
    for (double sig : unique_sig) {
        result.row_names.push_back(
            "sig_" + matrix_freq_helper::format_double(sig)
        );
    }
    for (size_t d = 0; d < n_dec; ++d) {
        for (size_t c = 0; c < n_conf; ++c) {
            std::string d_str = matrix_freq_helper::format_double(unique_dec[d]);
            if (conf == nullptr) {
                result.col_names.push_back("dec_" + d_str);
            } else {
                std::string c_str = matrix_freq_helper::format_double(
                    unique_conf[c]
                );
                result.col_names.push_back(
                    "dec_" + d_str + "_conf_" + c_str
                );
            }
        }
    }

    return result;
}