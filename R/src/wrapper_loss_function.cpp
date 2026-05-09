#include <Rcpp.h>
#include "Cpp/include/loss_function.hpp"

// 使用宏定义包住 include，骗过 Rcpp::sourceCpp 的正则检查
#define CORE_IMPL "Cpp/src/loss_function.cpp"
#include CORE_IMPL

using namespace Rcpp;

//' Calculate Model Loss (NLL, AIC, BIC)
// [[Rcpp::export]]
List loss_function(NumericMatrix mult_mat, NumericMatrix freq_mat, List params) {
    int n_rows = mult_mat.nrow();
    int n_cols = mult_mat.ncol();
    
    std::vector<std::vector<double>> cpp_mult(n_rows, std::vector<double>(n_cols));
    std::vector<std::vector<double>> cpp_freq(n_rows, std::vector<double>(n_cols));
    
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            cpp_mult[i][j] = mult_mat(i, j);
            cpp_freq[i][j] = freq_mat(i, j);
        }
    }
    
    int k = 0;
    if (params.containsElementNamed("numb_free")) {
        k = as<int>(params["numb_free"]);
    } else {
        stop("Error: 'params' must contain 'numb_free'. Please pass the output of modify_params().");
    }

    LossResult res;
    try {
        res = ::loss_function(cpp_mult, cpp_freq, k);
    } catch (std::exception& e) {
        stop(e.what());
    }

    // 返回给 R 一个包含所有计算结果的优雅 List
    return List::create(
        Named("logL") = res.logL,
        Named("nll") = res.nll,
        Named("k") = res.k,
        Named("aic") = res.aic,
        Named("bic") = res.bic
    );
}