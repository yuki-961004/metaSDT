#include <Rcpp.h>
#include "../../Cpp/include/matrix_freq.hpp"

#define CORE_IMPL "../../Cpp/src/matrix_freq.cpp"
#include CORE_IMPL
//' Calculate the frequency matrix for Signal Detection Theory (C++ core)
//'
//' @param stim The numeric vector indicating the signal type.
//' @param resp The numeric vector indicating the decision or response.
//' @param conf The numeric vector indicating the confidence level (optional).
//' @param diff The numeric vector indicating the difficulty level (optional).
//' @import Rcpp
//' @export
// [[Rcpp::export(name = "matrix_freq")]]
Rcpp::List r_matrix_freq(
  Rcpp::NumericVector stim, Rcpp::NumericVector resp, 
  Rcpp::Nullable<Rcpp::NumericVector> conf = R_NilValue,
  Rcpp::Nullable<Rcpp::NumericVector> diff = R_NilValue
) {
    std::vector<double> cpp_stim = Rcpp::as<std::vector<double>>(stim);
    std::vector<double> cpp_resp = Rcpp::as<std::vector<double>>(resp);

    std::vector<double> cpp_conf;
    const std::vector<double>* conf_ptr = nullptr;
    if (conf.isNotNull()) {
        cpp_conf = Rcpp::as<std::vector<double>>(conf);
        conf_ptr = &cpp_conf;
    }

    std::vector<double> cpp_diff;
    const std::vector<double>* diff_ptr = nullptr;
    if (diff.isNotNull()) {
        cpp_diff = Rcpp::as<std::vector<double>>(diff);
        diff_ptr = &cpp_diff;
    }

    MatrixFreq res;
    try {
        res = ::matrix_freq(
            /*stim=*/cpp_stim, /*resp=*/cpp_resp, /*conf=*/conf_ptr, /*diff=*/diff_ptr, /*std_params=*/nullptr
        );
    } catch (std::exception& e) {
    }

    int n_diffs = res.freq_mat.size();
    int n_rows = (n_diffs > 0) ? res.freq_mat[0].size() : 0;
    int out_cols = (n_rows > 0) ? res.freq_mat[0][0].size() : 0;
    
    Rcpp::List out_list;
    for (int d = 0; d < n_diffs; ++d) {
        Rcpp::NumericMatrix out_mat(n_rows, out_cols);
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < out_cols; ++j) {
                out_mat(i, j) = res.freq_mat[d][i][j];
            }
        }
        out_mat.attr("dimnames") = Rcpp::List::create(res.row_names, res.col_names);
        out_list.push_back(out_mat);
    }
    
    out_list.names() = res.dim_names;
    return out_list;
}
