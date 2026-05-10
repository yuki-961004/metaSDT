#include <Rcpp.h>
#include "Cpp/include/loss_function.hpp"
#include "Cpp/include/matrix_mult.hpp"

// 使用宏定义包住 include，骗过 Rcpp::sourceCpp 的正则检查
#define CORE_IMPL "Cpp/src/loss_function.cpp"
#include CORE_IMPL

using namespace Rcpp;

//' Calculate Model Loss (NLL, AIC, BIC)
// [[Rcpp::export(name = "loss_function")]]
List r_loss_function(NumericMatrix freq_mat, NumericMatrix prob_mat, List params) {
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
    
    int k = 0;
    if (params.containsElementNamed("numb_free")) {
        k = as<int>(params["numb_free"]);
    } else {
        stop("Error: 'params' must contain 'numb_free'. Please pass the output of modify_params().");
    }

    // 提取可能的极小值容差参数 (防止矩阵乘法时 log(0) 报错)
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    if (params.containsElementNamed("flat")) {
        List flat = params["flat"];
        if (flat.containsElementNamed("calc_tol")) {
            cpp_params["calc_tol"] = as<std::vector<double>>(flat["calc_tol"]);
        }
    }

    LossResult res;
    try {
        // 在 Wrapper 内部先执行矩阵乘法 (Freq * log(Prob))，彻底贴合 R 用户习惯
        auto cpp_mult = ::matrix_mult(cpp_freq, cpp_prob, cpp_params);
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