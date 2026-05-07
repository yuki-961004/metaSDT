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
    
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    std::vector<std::string> free_params;

    if (params.size() > 0 && params.hasAttribute("names")) {
        CharacterVector names = params.names();
        for (int i = 0; i < params.size(); ++i) {
            std::string key = as<std::string>(names[i]);
            if (key == "free_params") {
                free_params = as<std::vector<std::string>>(params[i]);
                continue;
            }
            if (key == "fixed_params" || key == "constant_params") continue;
            cpp_params[key] = as<std::vector<double>>(params[i]);
        }
    }

    LossResult res;
    try {
        res = ::loss_function(cpp_mult, cpp_freq, cpp_params, free_params);
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