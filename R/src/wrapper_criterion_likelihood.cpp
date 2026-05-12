#include <Rcpp.h>
#include "../../Cpp/include/criterion_likelihood.hpp"
#include "../../Cpp/include/matrix_mult.hpp"


using namespace Rcpp;

//' Calculate Model Likelihood indicators (NLL, AIC, BIC)
//' @export
// [[Rcpp::export(name = "criterion_likelihood")]]
List r_criterion_likelihood(List freq_mat, List prob_mat, List std_params) {
    int n_dims = freq_mat.size();
    NumericMatrix first = as<NumericMatrix>(freq_mat[0]);
    int n_rows = first.nrow();
    int n_cols = first.ncol();

    std::vector<std::vector<std::vector<double>>> cpp_freq(n_dims, std::vector<std::vector<double>>(n_rows, std::vector<double>(n_cols)));
    std::vector<std::vector<std::vector<double>>> cpp_prob(n_dims, std::vector<std::vector<double>>(n_rows, std::vector<double>(n_cols)));

    for (int d = 0; d < n_dims; ++d) {
        NumericMatrix f = as<NumericMatrix>(freq_mat[d]);
        NumericMatrix p = as<NumericMatrix>(prob_mat[d]);
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                cpp_freq[d][i][j] = f(i, j);
                cpp_prob[d][i][j] = p(i, j);
            }
        }
    }

    std::unordered_map<std::string, std::vector<double>> cpp_params;
    int k = 0;
    
    if (std_params.size() > 0 && std_params.hasAttribute("names")) {
        if (std_params.containsElementNamed("numb_free")) {
            k = as<int>(std_params["numb_free"]);
        } else {
            stop("Error: 'params' must contain 'numb_free'.");
        }
        CharacterVector names = std_params.names();
        for (int i = 0; i < std_params.size(); ++i) {
            std::string key = as<std::string>(names[i]);
            if (key != "name_free" && key != "name_fixed" && key != "name_constant" && 
                key != "numb_free" && key != "numb_fixed" && key != "numb_constant" &&
                key != "free_params" && key != "fixed_params" && key != "constant_params") {
                cpp_params[key] = as<std::vector<double>>(std_params[i]);
            }
        }
        if (std_params.containsElementNamed("flat")) {
            List flat = as<List>(std_params["flat"]);
            if (flat.containsElementNamed("calc_tol")) cpp_params["calc_tol"] = as<std::vector<double>>(flat["calc_tol"]);
        }
    } else {
        stop("Error: 'params' must contain 'numb_free'.");
    }

    auto cpp_mult = ::matrix_mult<double>(cpp_freq, cpp_prob, cpp_params);
    std::vector<double> free_params; 
    
    auto res = ::criterion_likelihood<double>(cpp_mult, cpp_freq, k, free_params, cpp_params);

    return List::create(
        Named("logL") = res.logL, Named("nll") = res.nll, Named("k") = res.k,
        Named("aic") = res.aic, Named("bic") = res.bic
    );
}