#ifndef MATRIX_FREQ_HPP
#define MATRIX_FREQ_HPP

#include <vector>
#include <string>
#include <unordered_map>

// 定义返回结果的结构体，非常易于跨语言绑定
struct MatrixFreq {
    std::vector<std::vector<std::vector<double>>> freq_mat; // 升级为 3D 数组: [dim][stim][resp]
    std::vector<std::string> dim_names;
    std::vector<std::string> row_names;
    std::vector<std::string> col_names;
};

// 主函数：计算频数矩阵
MatrixFreq matrix_freq(
    const std::vector<double>& stim,
    const std::vector<double>& resp,
    const std::vector<double>* conf = nullptr,
    const std::vector<double>* diff = nullptr,
    const std::unordered_map<std::string, std::vector<double>>* std_params = nullptr
);

#endif // MATRIX_FREQ_HPP