#include <Rcpp.h>
#include "../../Cpp/include/criterion_likelihood.hpp"
#include "../../Cpp/include/matrix_mult.hpp"

using namespace Rcpp;

//' Calculate Model Loss indicators (NLL, AIC, BIC)
//' @export
// [[Rcpp::export(name = "loss_function")]]
List r_loss_function(NumericMatrix freq_mat, NumericMatrix prob_mat, List params) {
    int n_rows = freq_mat.nrow();
    int n_cols = freq_mat.ncol();

    // 1. 数据转换
    std::vector<std::vector<double>> cpp_freq(n_rows, std::vector<double>(n_cols));
    std::vector<std::vector<double>> cpp_prob(n_rows, std::vector<double>(n_cols));

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            cpp_freq[i][j] = freq_mat(i, j);
            cpp_prob[i][j] = prob_mat(i, j);
        }
    }

    // 2. 提取参数与自由参数个数 k
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    int k = 0;
    
    if (params.size() > 0 && params.hasAttribute("names")) {
        if (params.containsElementNamed("numb_free")) {
            k = as<int>(params["numb_free"]);
        } else {
            stop("Error: 'params' must contain 'numb_free'.");
        }
    
        CharacterVector names = params.names();
        for (int i = 0; i < params.size(); ++i) {
            std::string key = as<std::string>(names[i]);
            if (key != "name_free" && key != "name_fixed" && key != "name_constant" && 
                key != "numb_free" && key != "numb_fixed" && key != "numb_constant" &&
                key != "free_params" && key != "fixed_params" && key != "constant_params") {
                cpp_params[key] = as<std::vector<double>>(params[i]);
            }
        }
        
        // 提取防止矩阵乘法时 log(0) 报错的计算容差 calc_tol
        if (params.containsElementNamed("flat")) {
            List flat = as<List>(params["flat"]);
            if (flat.containsElementNamed("calc_tol")) {
                cpp_params["calc_tol"] = as<std::vector<double>>(flat["calc_tol"]);
            }
        }
    } else {
        stop("Error: 'params' must contain 'numb_free'.");
    }

    // 3. 在底层 C++ 先执行矩阵乘法，再计算信息准则
    auto cpp_mult = ::matrix_mult(cpp_freq, cpp_prob, cpp_params);
    std::vector<double> free_params; // 预留空的探测数组，等同于单步调用时不激活惩罚项计算
    
    auto res = ::criterion_likelihood(cpp_mult, cpp_freq, k, free_params, cpp_params);

    // 4. 返回 R 语言 List
    return List::create(
        Named("logL") = res.logL,
        Named("nll") = res.nll,
        Named("k") = res.k,
        Named("aic") = res.aic,
        Named("bic") = res.bic
    );
}