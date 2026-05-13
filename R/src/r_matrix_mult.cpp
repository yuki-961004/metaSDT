#include <Rcpp.h>
#include "../../Cpp/include/matrix_mult.hpp"

#define CORE_IMPL "../../Cpp/src/matrix_mult.cpp"
#include CORE_IMPL
//' Calculate the Log-Likelihood product matrix
//' @export
// [[Rcpp::export(name = "matrix_mult")]]
Rcpp::List r_matrix_mult(Rcpp::List freq_mat, Rcpp::List prob_mat, Rcpp::List std_params) {
    int n_dims = freq_mat.size();
    Rcpp::NumericMatrix first = Rcpp::as<Rcpp::NumericMatrix>(freq_mat[0]);
    int n_rows = first.nrow();
    int n_cols = first.ncol();

    std::vector<std::vector<std::vector<double>>> cpp_freq(n_dims, std::vector<std::vector<double>>(n_rows, std::vector<double>(n_cols)));
    std::vector<std::vector<std::vector<double>>> cpp_prob(n_dims, std::vector<std::vector<double>>(n_rows, std::vector<double>(n_cols)));

    for (int d = 0; d < n_dims; ++d) {
        Rcpp::NumericMatrix f = Rcpp::as<Rcpp::NumericMatrix>(freq_mat[d]);
        Rcpp::NumericMatrix p = Rcpp::as<Rcpp::NumericMatrix>(prob_mat[d]);
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                cpp_freq[d][i][j] = f(i, j);
                cpp_prob[d][i][j] = p(i, j);
            }
        }
    }

    std::unordered_map<std::string, std::vector<double>> cpp_params;
    if (std_params.size() > 0 && std_params.hasAttribute("names")) {
        Rcpp::CharacterVector names = std_params.names();
        for (int i = 0; i < std_params.size(); ++i) {
            std::string key = Rcpp::as<std::string>(names[i]);
            if (key != "name_free" && key != "name_fixed" && key != "name_constant" && 
                key != "numb_free" && key != "numb_fixed" && key != "numb_constant" &&
                key != "free_params" && key != "fixed_params" && key != "constant_params") {
                cpp_params[key] = Rcpp::as<std::vector<double>>(std_params[i]);
            }
        }
    }

    auto res = ::matrix_mult<double>(cpp_freq, cpp_prob, cpp_params);

    Rcpp::List out_list;
    for (int d = 0; d < n_dims; ++d) {
        Rcpp::NumericMatrix out_mat(n_rows, n_cols);
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
