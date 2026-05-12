#include <Rcpp.h>
#include "../../Cpp/include/matrix_mult.hpp"

// 使用宏定义包住 include，骗过 Rcpp::sourceCpp 的正则检查
// 将核心的 C++ 源码拉取到当前编译单元中
#define CORE_IMPL "../../Cpp/src/matrix_mult.cpp"
#include CORE_IMPL

using namespace Rcpp;

//' Calculate the Log-Likelihood product matrix
//' @export
// [[Rcpp::export(name = "matrix_mult")]]
List r_matrix_mult(List freq_mat, List prob_mat, List std_params) {
    int n_dims = freq_mat.size();
    NumericMatrix first = as<NumericMatrix>(freq_mat[0]);
    int n_rows = first.nrow();
    int n_cols = first.ncol();

    // 1. 将 R 的 List of NumericMatrix 转换为底层的 3D vector
    std::vector<std::vector<std::vector<double>>> cpp_freq(n_dims, std::vector<std::vector<double>>(n_rows, std::vector<double>(n_cols)));
    std::vector<std::vector<std::vector<double>>> cpp_prob(n_dims, std::vector<std::vector<double>>(n_rows, std::vector<double>(n_cols)));

    for (int d = 0; d < n_dims; ++d) {
        NumericMatrix f = as<NumericMatrix>(freq_mat[d]);
        NumericMatrix p = as<NumericMatrix>(prob_mat[d]);
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                cpp_freq[d][i][j] = f(i, j);
                cpp_prob[d][i][j] = p(i, j);
            }
        }
    }

    // 2. 转换参数字典并剔除元数据
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    if (std_params.size() > 0 && std_params.hasAttribute("names")) {
        CharacterVector names = std_params.names();
        for (int i = 0; i < std_params.size(); ++i) {
            std::string key = as<std::string>(names[i]);
            if (key != "name_free" && key != "name_fixed" && key != "name_constant" && 
                key != "numb_free" && key != "numb_fixed" && key != "numb_constant" &&
                key != "free_params" && key != "fixed_params" && key != "constant_params") {
                cpp_params[key] = as<std::vector<double>>(std_params[i]);
            }
        }
    }

    // 3. 调用核心 C++ 函数
    auto res = ::matrix_mult<double>(cpp_freq, cpp_prob, cpp_params);

    // 4. 将结果转换回 R 的 List of NumericMatrix 并保留原有的行列名
    List out_list;
    for (int d = 0; d < n_dims; ++d) {
        NumericMatrix out_mat(n_rows, n_cols);
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                out_mat(i, j) = res[d][i][j];
            }
        }
        out_mat.attr("dimnames") = first.attr("dimnames");
        out_list.push_back(out_mat);
    }
    out_list.names() = freq_mat.names();
    return out_list;
}