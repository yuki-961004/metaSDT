#ifndef MATRIX_FREQ_HPP
#define MATRIX_FREQ_HPP

#include <vector>
#include <string>

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