#include <Rcpp.h>
#include "Cpp/include/matrix_mult.hpp"

// 使用宏定义包住 include，骗过 Rcpp::sourceCpp 的正则检查
#define CORE_IMPL "Cpp/src/matrix_mult.cpp"
#include CORE_IMPL

using namespace Rcpp;

//' Calculate the Log-Likelihood product matrix
// [[Rcpp::export]]
NumericMatrix matrix_mult(NumericMatrix freq_mat, NumericMatrix prob_mat, List params) {
    // 1. 将 R 矩阵转换为 C++ 的 2D vector
    int n_rows = freq_mat.nrow();
    int n_cols = freq_mat.ncol();
    
    std::vector<std::vector<double>> cpp_freq(n_rows, std::vector<double>(n_cols));
    std::vector<std::vector<double>> cpp_prob(n_rows, std::vector<double>(n_cols));
    
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            cpp_freq[i][j] = freq_mat(i, j);
            cpp_prob[i][j] = prob_mat(i, j);
        }
    }
    
    // 2. 转换参数字典 (自动跳过分类元数据)
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    if (params.size() > 0 && params.hasAttribute("names")) {
        CharacterVector names = params.names();
        for (int i = 0; i < params.size(); ++i) {
            std::string key = as<std::string>(names[i]);
            if (key == "name_free" || key == "name_fixed" || 
                key == "name_constant" || key == "numb_free" || 
                key == "numb_fixed" || key == "numb_constant" ||
                key == "free_params" || key == "fixed_params" || 
                key == "constant_params") {
                continue;
            }
            cpp_params[key] = as<std::vector<double>>(params[i]);
        }
    }

    // 3. 调用 C++ 核心函数计算
    std::vector<std::vector<double>> res;
    try {
        res = ::matrix_mult(cpp_freq, cpp_prob, cpp_params);
    } catch (std::exception& e) {
        stop(e.what());
    }

    // 4. 将结果转回 R 的 NumericMatrix
    NumericMatrix out_mat(n_rows, n_cols);
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            out_mat(i, j) = res[i][j];
        }
    }
    
    // 5. 继承原矩阵的行列名
    if (freq_mat.hasAttribute("dimnames")) out_mat.attr("dimnames") = freq_mat.attr("dimnames");
    return out_mat;
}