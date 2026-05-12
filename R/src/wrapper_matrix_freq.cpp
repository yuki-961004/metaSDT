#include <Rcpp.h>
#include "../../Cpp/include/matrix_freq.hpp"

// 使用宏定义包住 include，骗过 Rcpp::sourceCpp 的正则检查
// 避免它自作主张将 cpp 文件单独抽出编译，从而解决"重复链接"和"多重定义"的问题
#define CORE_IMPL "../../Cpp/src/matrix_freq.cpp"
#include CORE_IMPL

using namespace Rcpp;

//' Calculate the frequency matrix for Signal Detection Theory (C++ core)
//'
//' @param stim The numeric vector indicating the signal type.
//' @param resp The numeric vector indicating the decision or response.
//' @param conf The numeric vector indicating the confidence level (optional).
//' @param diff The numeric vector indicating the difficulty level (optional).
//' @import Rcpp
//' @export
// [[Rcpp::export(name = "matrix_freq")]]
List r_matrix_freq(
  NumericVector stim, NumericVector resp, 
  Rcpp::Nullable<NumericVector> conf = R_NilValue,
  Rcpp::Nullable<NumericVector> diff = R_NilValue
) {
    // 1. 将 R 传入的向量(NumericVector)深度复制并转换为纯 C++ 能够直接识别和处理的 std::vector<double>
    std::vector<double> cpp_stim = as<std::vector<double>>(stim);
    std::vector<double> cpp_resp = as<std::vector<double>>(resp);

    std::vector<double> cpp_conf;
    const std::vector<double>* conf_ptr = nullptr;
    if (conf.isNotNull()) {
        cpp_conf = as<std::vector<double>>(conf);
        conf_ptr = &cpp_conf;
    }

    std::vector<double> cpp_diff;
    const std::vector<double>* diff_ptr = nullptr;
    if (diff.isNotNull()) {
        cpp_diff = as<std::vector<double>>(diff);
        diff_ptr = &cpp_diff;
    }

    // 2. 调用纯 C++ 底层的核心计算函数
    MatrixFreq res;
    try {
        res = ::matrix_freq(
            /*stim=*/cpp_stim, /*resp=*/cpp_resp, /*conf=*/conf_ptr, /*diff=*/diff_ptr, /*std_params=*/nullptr
        );
    } catch (std::exception& e) {
        stop(e.what()); // 捕获底层 C++ 抛出的 std::exception，并安全地转抛为 R 语言端的 Error
    }

    int n_diffs = res.freq_mat.size();
    int n_rows = (n_diffs > 0) ? res.freq_mat[0].size() : 0;
    int out_cols = (n_rows > 0) ? res.freq_mat[0][0].size() : 0;
    
    List out_list;
    for (int d = 0; d < n_diffs; ++d) {
        NumericMatrix out_mat(n_rows, out_cols);
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < out_cols; ++j) {
                out_mat(i, j) = res.freq_mat[d][i][j];
            }
        }
        out_mat.attr("dimnames") = List::create(res.row_names, res.col_names);
        out_list.push_back(out_mat);
    }
    
    out_list.names() = res.dim_names;
    return out_list;
}