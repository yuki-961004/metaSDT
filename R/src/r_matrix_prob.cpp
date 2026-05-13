#include <Rcpp.h>
#include "../../Cpp/include/matrix_prob.hpp"

#define CORE_IMPL "../../Cpp/src/matrix_prob.cpp"
#include CORE_IMPL
//' Calculate Probability Matrix
// [[Rcpp::export(name = "matrix_prob")]]
Rcpp::List r_matrix_prob(Rcpp::List cdf_noise, Rcpp::List cdf_signal, Rcpp::List params) {
    int n_dims = cdf_noise.size();
    std::vector<std::vector<double>> cpp_noise(n_dims);
    std::vector<std::vector<double>> cpp_signal(n_dims);
    for (int d = 0; d < n_dims; ++d) {
        cpp_noise[d] = Rcpp::as<std::vector<double>>(cdf_noise[d]);
        cpp_signal[d] = Rcpp::as<std::vector<double>>(cdf_signal[d]);
    }
    
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    if (params.size() > 0 && params.hasAttribute("names")) {
        Rcpp::CharacterVector names = params.names();
        for (int i = 0; i < params.size(); ++i) {
            std::string key = Rcpp::as<std::string>(names[i]);
            if (key == "name_free" || key == "name_fixed" || 
                key == "name_constant" || key == "numb_free" || 
                key == "numb_fixed" || key == "numb_constant" ||
                key == "free_params" || key == "fixed_params" || 
                key == "constant_params") {
                continue;
            }
            cpp_params[key] = Rcpp::as<std::vector<double>>(params[i]);
        }
    }

    MatrixProb<double> res = ::matrix_prob<double>(cpp_noise, cpp_signal, cpp_params);

    Rcpp::List out_list;
    n_dims = res.prob_mat.size();
    int n_rows = (n_dims > 0) ? res.prob_mat[0].size() : 0;
    int n_cols = (n_rows > 0) ? res.prob_mat[0][0].size() : 0;
    
    for (int d = 0; d < n_dims; ++d) {
        Rcpp::NumericMatrix out_mat(n_rows, n_cols);
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                out_mat(i, j) = res.prob_mat[d][i][j];
            }
        }
        out_mat.attr("dimnames") = Rcpp::List::create(res.row_names, res.col_names);
        out_list.push_back(out_mat);
    }
    out_list.names() = res.dim_names;
    return out_list;
}
