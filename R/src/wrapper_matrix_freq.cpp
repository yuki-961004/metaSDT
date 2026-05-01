#include <Rcpp.h>
#include "Cpp/include/matrix_freq.hpp"
// 纯 C++ 源文件目前已经放置在 R/src 内，继续使用联合编译
#include "Cpp/src/matrix_freq.cpp"

using namespace Rcpp;

//' Calculate the frequency matrix for Signal Detection Theory (C++ core)
//'
//' @param stim The numeric vector indicating the signal type.
//' @param resp The numeric vector indicating the decision or response.
//' @param conf The numeric vector indicating the confidence level (optional).
//' @import Rcpp
//' @export
// [[Rcpp::export]]
NumericMatrix matrix_freq(
  NumericVector stim, NumericVector resp, Rcpp::Nullable<NumericVector> conf = R_NilValue
) {
    // 1. 将 R 向量转换为纯 C++ 的 std::vector<double>
    std::vector<double> cpp_stim = as<std::vector<double>>(stim);
    std::vector<double> cpp_resp = as<std::vector<double>>(resp);

    std::vector<double> cpp_conf;
    const std::vector<double>* conf_ptr = nullptr;
    if (conf.isNotNull()) {
        cpp_conf = as<std::vector<double>>(conf);
        conf_ptr = &cpp_conf;
    }

    // 2. 调用纯 C++ 底层函数
    MatrixFreq res;
    try {
        res = ::matrix_freq(cpp_stim, cpp_resp, conf_ptr);
    } catch (std::exception& e) {
        stop(e.what()); // 将 C++ 异常转抛为 R 报错
    }

    // 3. 将 C++ 结果转换为 R 的 NumericMatrix
    int n_rows = res.freq_mat.size();
    int out_cols = (n_rows > 0) ? res.freq_mat[0].size() : 0;
    NumericMatrix out_mat(n_rows, out_cols);

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            out_mat(i, j) = res.freq_mat[i][j];
        }
    }

    // 4. 设置行列名并返回
    out_mat.attr("dimnames") = List::create(res.row_names, res.col_names);
    return out_mat;
}