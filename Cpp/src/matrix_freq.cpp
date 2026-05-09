#include "../include/matrix_freq.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace {
    // 匿名命名空间：内部辅助函数仅在当前文件内可见，不会引起符号冲突

    int find_index(const std::vector<double>& vec, double target) {
        // 使用标准库 std::find 在向量中查找目标值
        auto it = std::find(vec.begin(), vec.end(), target);
        if (it != vec.end()) {
            // 如果找到，计算迭代器与起始位置的距离，即为其索引值 (0-based)
            return static_cast<int>(std::distance(vec.begin(), it));
        }
        // 如果未找到返回 -1（但在本算法中，target 总是来源于 vec，所以理论上不会发生）
        return -1;
    }

    std::vector<double> get_unique_sorted(const std::vector<double>& vec) {
        // 利用 std::set 的特性：插入元素时会自动进行去重和升序排序
        std::set<double> s(vec.begin(), vec.end());
        // 将处理后的 std::set 重新转换为 std::vector，
        // 以便后续能够使用下标进行随机访问
        return std::vector<double>(s.begin(), s.end());
    }

    std::string format_double(double val) {
        // 先将浮点数转换为默认的字符串格式
        std::string s = std::to_string(val);
        // 查找最后一个非 '0' 的字符位置，并截断其后所有的尾随 '0'
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        // 如果截断 0 之后字符串末尾刚好是小数点，
        // 则一并去除 (例如 "1." 会被整理成 "1")
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
        // 第一步校验：刺激和反应的向量长度必须一致，以防后续按行遍历时出现越界
        if (resp.size() != n_rows) {
            throw std::invalid_argument(
                "Error: 'stim' and 'resp' must have the same length."
            );
        }

        // 预分配一个 N 行 3 列的二维矩阵，
        // 将分散的输入向量整合成结构化数据，方便后续统一按行处理
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

        // 逐行组装数据：第 0 列为 stim，第 1 列为 resp，第 2 列为 conf
        for (size_t i = 0; i < n_rows; ++i) {
            out_mat[i][0] = stim[i];
            out_mat[i][1] = resp[i];
            if (conf != nullptr) {
                out_mat[i][2] = (*conf)[i];
            } else {
                // 若未传入信心指数(conf)，则默认填充 1.0，
                // 从而在后续计算时退化为经典的信号检测论频数矩阵
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

    // ==========================================================
    // 1. 整理输入数据，合并成统一的数值矩阵
    // ==========================================================
    auto num_mat = prep_num_mat(
        stim, resp, conf
    );
    size_t n_rows = num_mat.size();

    // ==========================================================
    // 2. 拆分维度并提取唯一值 (Unique Values)
    // ==========================================================
    // 按列拆分数值矩阵，目的是针对各个维度提取并计算各自包含的唯一值
    std::vector<double> stim_col, resp_col, conf_col;
    for (size_t i = 0; i < n_rows; ++i) {
        stim_col.push_back(num_mat[i][0]);
        resp_col.push_back(num_mat[i][1]);
        conf_col.push_back(num_mat[i][2]);
    }

    // 提取各个维度的唯一值并自动排序
    auto unique_stim = get_unique_sorted(stim_col);
    auto unique_resp = get_unique_sorted(resp_col);
    auto unique_conf = get_unique_sorted(conf_col);

    // ==========================================================
    // 3. 计算并预分配结果矩阵
    // ==========================================================
    // 计算结果矩阵的行列大小：
    // 行数为不同刺激的数量，
    // 列数为 (不同反应的数量 × 不同信心指数的数量)
    size_t n_stim = unique_stim.size();
    size_t n_resp = unique_resp.size();
    size_t n_conf = unique_conf.size();
    size_t n_cols_out = n_resp * n_conf;

    // 预分配结果对象，并将其包含的二维频数矩阵用 0.0 填满初始化
    MatrixFreq result;
    result.freq_mat.assign(
        n_stim, std::vector<double>(n_cols_out, 0.0)
    );

    // ==========================================================
    // 4. 遍历并填充频数矩阵
    // ==========================================================
    // 遍历原始数据的每一行，查找其在 unique 集合中的索引，
    // 进而定位并累加到结果矩阵的正确位置
    for (size_t i = 0; i < n_rows; ++i) {
        int row_idx = find_index(
            unique_stim, num_mat[i][0]
        );
        int resp_idx = find_index(
            unique_resp, num_mat[i][1]
        );
        int conf_idx = find_index(
            unique_conf, num_mat[i][2]
        );

        // 将反应与信心的二维关系展平为一维列索引
        // 按照 321123 的自然顺序:
        // 针对噪声反应 (resp_idx == 0) 置信度倒序，其余正序
        int col_idx;
        if (resp_idx == 0) {
            col_idx = (n_conf - 1) - conf_idx;
        } else {
            col_idx = resp_idx * n_conf + conf_idx;
        }
        // 在对应的矩阵单元格累加频数 (+1)
        result.freq_mat[row_idx][col_idx] += 1.0; 
    }

    // ==========================================================
    // 5. 生成直观的行列元数据 (Row / Col Names)
    // ==========================================================
    for (double stim_val : unique_stim) {
        result.row_names.push_back(
            "stim_" + format_double(stim_val)
        );
    }
    for (size_t r = 0; r < n_resp; ++r) {
        for (size_t c_loop = 0; c_loop < n_conf; ++c_loop) {
            // 根据自然排序，resp 0 的 confidence 倒序排列，其余正序
            size_t c = (r == 0) ? (n_conf - 1 - c_loop) : c_loop;
            std::string r_str = format_double(unique_resp[r]);
            if (conf == nullptr) {
                // 若没有使用信心指数，则列名仅标记反应类型
                result.col_names.push_back("resp_" + r_str);
            } else {
                std::string c_str = format_double(
                    unique_conf[c]
                );
                result.col_names.push_back(
                    // 若有信心指数，列名格式如：resp_1_conf_2
                    "resp_" + r_str + "_conf_" + c_str
                );
            }
        }
    }

    return result;
}