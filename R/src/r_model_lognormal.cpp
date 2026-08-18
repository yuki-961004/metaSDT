#include <Rcpp.h>
#include <metaSDT/model_lognormal.hpp>

#define CORE_IMPL "cpp/model_lognormal.cpp"
#include CORE_IMPL

// [[Rcpp::export(name = ".core_model_lognormal")]]
Rcpp::List r_model_lognormal(Rcpp::List params) {
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
    ModelLognormal<double> model(cpp_params);
    
    return Rcpp::List::create(
        Rcpp::Named("prob_mat") = model.compute_probabilities(),
        Rcpp::Named("criteria") = model.get_criteria()
    );
}
