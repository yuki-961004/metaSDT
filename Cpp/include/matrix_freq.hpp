#ifndef MATRIX_FREQ_HPP
#define MATRIX_FREQ_HPP

#include <vector>
#include <string>

namespace matrix_freq_helper {
    // 辅助函数：寻找向量中目标值的索引 (0-based)
    int find_index(const std::vector<double>& vec, double target);

    // 辅助函数：获取排序后的唯一值集合
    std::vector<double> get_unique_sorted(const std::vector<double>& vec);

    // 辅助函数：格式化浮点数为字符串并去除多余后缀零 (用于行列名)
    std::string format_double(double val);

    // 辅助函数：提取目标列并构建纯数值型矩阵
    std::vector<std::vector<double>> prep_num_mat(
        const std::vector<double>& stim,
        const std::vector<double>& resp,
        const std::vector<double>* conf = nullptr
    );
}

// 定义返回结果的结构体，非常易于跨语言绑定
struct MatrixFreq {
    std::vector<std::vector<double>> freq_mat;
    std::vector<std::string> row_names;
    std::vector<std::string> col_names;
};

// 主函数：计算频数矩阵
MatrixFreq matrix_freq(
    const std::vector<double>& stim,
    const std::vector<double>& resp,
    const std::vector<double>* conf = nullptr
);

#endif // MATRIX_FREQ_HPP