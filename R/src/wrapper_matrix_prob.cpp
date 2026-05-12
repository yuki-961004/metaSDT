#include <Rcpp.h>
#include "../../Cpp/include/matrix_prob.hpp"

// 使用宏定义解决多重编译问题
#define CORE_IMPL "../../Cpp/src/matrix_prob.cpp"
#include CORE_IMPL

using namespace Rcpp;

//' Calculate Probability Matrix
// [[Rcpp::export(name = "matrix_prob")]]
List r_matrix_prob(List cdf_noise, List cdf_signal, List params) {
    // 1. 转换 2D 向量
    int n_dims = cdf_noise.size();
    std::vector<std::vector<double>> cpp_noise(n_dims);
    std::vector<std::vector<double>> cpp_signal(n_dims);
    for (int d = 0; d < n_dims; ++d) {
        cpp_noise[d] = as<std::vector<double>>(cdf_noise[d]);
        cpp_signal[d] = as<std::vector<double>>(cdf_signal[d]);
    }
    
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
    MatrixProb<double> res = ::matrix_prob<double>(cpp_noise, cpp_signal, cpp_params);

    // 4. 将 3D std::vector 转换为 R 的 List of NumericMatrix
    List out_list;
    n_dims = res.prob_mat.size();
    int n_rows = (n_dims > 0) ? res.prob_mat[0].size() : 0;
    int n_cols = (n_rows > 0) ? res.prob_mat[0][0].size() : 0;
    
    for (int d = 0; d < n_dims; ++d) {
        NumericMatrix out_mat(n_rows, n_cols);
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                out_mat(i, j) = res.prob_mat[d][i][j];
            }
        }
        out_mat.attr("dimnames") = List::create(res.row_names, res.col_names);
        out_list.push_back(out_mat);
    }
    out_list.names() = res.dim_names;
    return out_list;
}