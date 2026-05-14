#include "../include/matrix_freq.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <cmath>

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
        const std::vector<double>* conf,
        const std::vector<double>* diff) {

        size_t n_rows = stim.size();
        // 第一步校验：刺激和反应的向量长度必须一致，以防后续按行遍历时出现越界
        if (resp.size() != n_rows) {
            throw std::invalid_argument(
                "Error: 'stim' and 'resp' must have the same length."
            );
        }

        // 预分配一个 N 行 4 列的二维矩阵，
        // 将分散的输入向量整合成结构化数据，方便后续统一按行处理
        std::vector<std::vector<double>> out_mat(
            n_rows, std::vector<double>(4, 0.0)
        );

        if (conf != nullptr) {
            if (conf->size() != n_rows) {
                throw std::invalid_argument(
                    "Error: 'conf' must have the same length as "
                    "'stim' and 'resp'."
                );
            }
        }
        if (diff != nullptr && diff->size() != n_rows) {
            throw std::invalid_argument(
                "Error: 'diff' must have the same length."
            );
        }

        // 逐行组装数据：第 0 列为 stim，第 1 列为 resp，第 2 列为 conf，第 3 列为 diff
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
            if (diff != nullptr) {
                out_mat[i][3] = (*diff)[i];
            } else {
                out_mat[i][3] = 1.0; 
            }
        }
        return out_mat;
    }

    // ////////////////////////////////////////////////////////////////////////
    // 智能处理置信度数据 (Process Confidence Data)
    // 1. 检查数据与模型参数是否匹配。
    // 2. 如果传入的是连续数据（如滑块评分），而模型需要离散等级，则自动分箱。
    // 3. 如果传入的是离散数据，但与模型期望的等级数不符，则抛出错误。
    // ////////////////////////////////////////////////////////////////////////
    void process_confidence_ratings(
        const std::vector<double>* conf,
        const std::unordered_map<std::string, std::vector<double>>* std_params,
        std::vector<double>& processed_conf,
        const std::vector<double>*& final_conf_ptr
    ) {
        if (
            conf == nullptr || 
            std_params == nullptr || 
            !std_params->count("n_conf") || 
            std_params->at("n_conf").empty()
        ) {
            return;
        }

        // std_params 中的 n_conf 代表判定标准的总数量 (Criteria)
        // 每一个反应选项下的信心等级数 (num_bins) = (n_conf + 1) / 2
        int n_criteria = static_cast<int>(std_params->at("n_conf")[0]);
        int num_bins = (n_criteria + 1) / 2;
        
        if (num_bins <= 1) {
            return;
        }

        auto unique_conf = get_unique_sorted(*conf);
        // 如果唯一的置信度取值数量不多于模型期望的等级数，则无需处理
        if (unique_conf.size() <= static_cast<size_t>(num_bins)) {
            return;
        }

        double min_c = unique_conf.front();
        double max_c = unique_conf.back();
        
        // 1. 智能探针：判断是否为离散等级 (Discrete Ordinal) 或 连续变量 (Continuous)
        bool is_integer = true;
        for (double v : unique_conf) {
            // 允许极小的浮点误差，若含有小数即判定为非纯整数
            if (std::abs(v - std::round(v)) > 1e-6) {
                is_integer = false;
                break;
            }
        }
        
        double span = max_c - min_c + 1.0;
        // 如果全是整数，且最大值与最小值的跨度处于常规等级量表范围内(<= 25)，则判定为纯离散等级变量
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
                "If you intend to fit " + std::to_string(unique_conf.size()) + 
                " confidence levels, you must provide exactly " + 
                std::to_string(expected_c_conf) + " 'c_conf' boundaries.\n"
                "(Note: If your raw data is a 1-N rating scale, ensure it is "
                "split into a binary 'resp' column and a magnitude 'conf' "
                "column first)."
            );
        }

        // 2. 如果判定为连续滑块变量 (Continuous)，则自动执行线性等距分箱 (Binning)
        if (max_c > min_c) {
            processed_conf.reserve(conf->size());
            for (double c : *conf) {
                // 线性缩放到 [0, 1] 后映射到 1 到 num_bins
                double norm_c = (c - min_c) / (max_c - min_c);
                int bin = static_cast<int>(std::floor(norm_c * num_bins)) + 1;
                if (bin > num_bins) bin = num_bins; // 防止极端值溢出
                processed_conf.push_back(static_cast<double>(bin));
            }
            final_conf_ptr = &processed_conf;
        }
    }
}

MatrixFreq matrix_freq(
    const std::vector<double>& stim,
    const std::vector<double>& resp,
    const std::vector<double>* conf,
    const std::vector<double>* diff,
    const std::unordered_map<std::string, std::vector<double>>* std_params) {

    std::vector<double> processed_conf;
    const std::vector<double>* final_conf_ptr = conf;

    // ////////////////////////////////////////////////////////////////////////
    // 0. 连续评分智能分箱 (Smart Continuous Discretization)
    // 这是一个极具鲁棒性的“安检+转换”门。
    // 如果用户传入的是连续信心评分 (例如 0~100 的滑块，或纯正的反应时 RT)，
    // 但模型设定了固定的离散等级 (n_conf)，这里会自动按比例将其线性映射并分箱。
    // 甚至它还能检测离散输入是否与模型参数完全匹配，不匹配则抛出极具指导意义的 Error！
    // ////////////////////////////////////////////////////////////////////////
    process_confidence_ratings(
        /*conf=*/conf, /*std_params=*/std_params, 
        /*processed_conf=*/processed_conf, /*final_conf_ptr=*/final_conf_ptr
    );

    // ////////////////////////////////////////////////////////////////////////
    // 1. 试次数据“拉链”组装 (Zipping Vectors into a Matrix)
    // 从 R/Python 传进来的数据是四根散装的平行向量 (stim, resp, conf, diff)。
    // 就像把四根并排的毛线拧成一股绳，我们在这里把它们打包成一个 N行 × 4列 的“长表”。
    // 这样在后续 C++ 底层遍历时，我们能以极高的 CPU 缓存命中率逐行读取数据。
    // ////////////////////////////////////////////////////////////////////////
    auto num_mat = prep_num_mat(
        /*stim=*/stim, /*resp=*/resp, /*conf=*/final_conf_ptr, /*diff=*/diff
    );
    size_t n_rows = num_mat.size();

    // ////////////////////////////////////////////////////////////////////////
    // 2. 维度探测与唯一值提取 (Dimension Probing)
    // 提取这 N 次作答中，到底出现了哪几种刺激、哪几种反应、哪几种置信度。
    // 获取到的这些唯一值的数量 (unique_xxx.size())，将直接决定
    // 我们下一步要生成的终极 3D 频数汇总矩阵的“长、宽、高”。
    // ////////////////////////////////////////////////////////////////////////
    // 按列拆分数值矩阵，目的是针对各个维度提取并计算各自包含的唯一值
    std::vector<double> stim_col, resp_col, conf_col, diff_col;
    for (size_t i = 0; i < n_rows; ++i) {
        stim_col.push_back(num_mat[i][0]);
        resp_col.push_back(num_mat[i][1]);
        conf_col.push_back(num_mat[i][2]);
        diff_col.push_back(num_mat[i][3]);
    }

    // 提取各个维度的唯一值并自动排序
    auto unique_stim = get_unique_sorted(stim_col);
    auto unique_resp = get_unique_sorted(resp_col);
    auto unique_conf = get_unique_sorted(conf_col);
    auto unique_diff = get_unique_sorted(diff_col);

    // If model parameters specify expected confidence bins, lock matrix columns
    // to that schema so missing observed bins become zero-count columns instead
    // of shrinking freq_mat dimensions and causing downstream mismatch.
    int expected_conf_bins = -1;
    if (std_params != nullptr && std_params->count("n_conf") && 
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

    // ////////////////////////////////////////////////////////////////////////
    // 3. 终极频数汇总矩阵预分配 (Matrix Allocation)
    // 这里的 result.freq_mat 就是信号检测论里用于 MLE 拟合的核心矩阵！
    // [高/层] n_diffs：难度等级 
    // [宽/行] n_stim： 刺激类型 (通常是 2 行：信号与噪声)
    // [长/列] n_cols_out： 反应键 × 置信度选项 (如 2键×3信心 = 6 列)
    // 我们提前分配好三维矩阵，并用 0.0 把所有格子填满，准备开始记账。
    // ////////////////////////////////////////////////////////////////////////
    // 计算结果矩阵的行列大小：
    // 行数为不同刺激的数量，
    // 列数为 (不同反应的数量 × 不同信心指数的数量)
    size_t n_stim = unique_stim.size();
    size_t n_resp = unique_resp.size();
    size_t n_conf = unique_conf.size();
    size_t n_diffs = unique_diff.size();
    size_t n_cols_out = n_resp * n_conf;

    // 预分配结果对象，并将其包含的二维频数矩阵用 0.0 填满初始化
    MatrixFreq result;
    result.freq_mat.assign(
        n_diffs, std::vector<std::vector<double>>(
            n_stim,
            std::vector<double>(n_cols_out, 0.0)
        )
    );

    // ////////////////////////////////////////////////////////////////////////
    // 4. 核心记账引擎：遍历试次并累加频数 (Tallying loop)
    // 扫描 N 次实验记录。最精妙的是 col_idx 的一维折叠计算：
    // 它把 2D 的反应和置信度，精准地按照 SDT 的经典 "321|123" 对称模式，
    // 折叠压平到了这一维度的列索引里。定位到正确格子后，无情地 +1.0 ！
    // ////////////////////////////////////////////////////////////////////////
    // 遍历原始数据的每一行，查找其在 unique 集合中的索引，
    // 进而定位并累加到结果矩阵的正确位置
    for (size_t i = 0; i < n_rows; ++i) {
        int row_idx = find_index(
            /*vec=*/unique_stim, /*target=*/num_mat[i][0]
        );
        int resp_idx = find_index(
            /*vec=*/unique_resp, /*target=*/num_mat[i][1]
        );
        int conf_idx = find_index(
            /*vec=*/unique_conf, /*target=*/num_mat[i][2]
        );
        if (conf_idx < 0 && expected_conf_bins > 1) {
            double c = num_mat[i][2];
            // Fallback for unusual coding: linearly project to expected bins.
            int mapped = static_cast<int>(std::round(c));
            if (mapped < 1) mapped = 1;
            if (mapped > expected_conf_bins) mapped = expected_conf_bins;
            conf_idx = mapped - 1;
        }
        int diff_idx = find_index(
            /*vec=*/unique_diff, /*target=*/num_mat[i][3]
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
        // 在 3D 矩阵单元格累加频数
        result.freq_mat[diff_idx][row_idx][col_idx] += 1.0; 
    }

    // ////////////////////////////////////////////////////////////////////////
    // 5. 元数据贴标 (Metadata Assembling)
    // 纯 C++ 的 std::vector 是没有行列名的，直接传回 R/Python 是一本糊涂账。
    // 所以我们在这里显式生成如 "stim_1", "resp_0_conf_3" 的字符串贴图。
    // 当它穿越绑定层回到外层语言时，就会被自动变为带优雅表头的 DataFrame！
    // ////////////////////////////////////////////////////////////////////////
    for (double diff_val : unique_diff) {
        result.dim_names.push_back("diff_" + format_double(diff_val));
    }
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
