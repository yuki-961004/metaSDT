#include <Rcpp.h>
#include <matrix_prob.hpp>

// 使用宏定义解决多重编译问题
#define CORE_IMPL "Cpp/src/matrix_prob.cpp"
#include CORE_IMPL

using namespace Rcpp;

//' Calculate Probability Matrix
// [[Rcpp::export]]
NumericMatrix matrix_prob(NumericVector cdf_noise, NumericVector cdf_signal, List params) {
    // 1. 转换向量
    std::vector<double> cpp_noise = as<std::vector<double>>(cdf_noise);
    std::vector<double> cpp_signal = as<std::vector<double>>(cdf_signal);
    
    // 2. 转换参数字典
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    if (params.size() > 0 && params.hasAttribute("names")) {
        CharacterVector names = params.names();
        for (int i = 0; i < params.size(); ++i) {
            std::string key = as<std::string>(names[i]);
            // 拦截包含元数据的槽位，防止将其强制转换为 double 数组时报错
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

    // 3. 调用核心 C++ 函数
    MatrixProb res = ::matrix_prob(cpp_noise, cpp_signal, cpp_params);

    // 4. 将 2D std::vector 优雅地转换为 R 的 NumericMatrix
    NumericMatrix out_mat(res.prob_mat.size(), res.prob_mat[0].size());
    for (size_t i = 0; i < res.prob_mat.size(); ++i) {
        for (size_t j = 0; j < res.prob_mat[0].size(); ++j) {
            out_mat(i, j) = res.prob_mat[i][j];
        }
    }

    // 5. 赋予 R 语言中矩阵原生的 dimnames (行列名)
    out_mat.attr("dimnames") = List::create(res.row_names, res.col_names);
    
    return out_mat;
}