// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <Eigen/Dense>
#include "../../Cpp/include/modify_params.hpp"
#include "../../Cpp/include/modify_prior.hpp"
#include "../../Cpp/include/criterion_prior.hpp"
namespace {
    void r_obj_to_cpp_map(SEXP r_obj, std::unordered_map<std::string, std::vector<double>>& cpp_map) {
        if (Rf_isNull(r_obj) || Rf_length(r_obj) == 0) return;
        Rcpp::RObject robj(r_obj);
        if (!robj.hasAttribute("names")) return;
        Rcpp::CharacterVector names = robj.attr("names");
        if (names.isNULL() || Rf_length(names) != Rf_length(robj)) return;
        if (Rcpp::is<Rcpp::List>(robj)) {
            Rcpp::List r_list(robj);
            for (int i = 0; i < r_list.size(); ++i) {
                cpp_map[Rcpp::as<std::string>(names[i])] = Rcpp::as<std::vector<double>>(r_list[i]);
            }
        } else if (Rcpp::is<Rcpp::NumericVector>(robj)) {
            Rcpp::NumericVector r_vec(robj);
            for (int i = 0; i < r_vec.size(); ++i) {
                cpp_map[Rcpp::as<std::string>(names[i])] = { r_vec[i] };
            }
        }
    }
}

//' Evaluate Log-Prior
//' @export
// [[Rcpp::export(name = "criterion_prior")]]
double r_criterion_prior(Rcpp::List user_priors, Rcpp::RObject std_params = R_NilValue) {
    ModifiedParamsResult param_info;
    if (!std_params.isNULL() && Rf_length(std_params) > 0) {
        if (Rcpp::is<Rcpp::List>(std_params)) {
            Rcpp::List r_list(std_params);
            if (r_list.containsElementNamed("name_free")) {
                param_info.name_free = Rcpp::as<std::vector<std::string>>(r_list["name_free"]);
                if (r_list.containsElementNamed("name_fixed")) param_info.name_fixed = Rcpp::as<std::vector<std::string>>(r_list["name_fixed"]);
                if (r_list.containsElementNamed("name_constant")) param_info.name_constant = Rcpp::as<std::vector<std::string>>(r_list["name_constant"]);
                for (const auto& name : param_info.name_free) param_info.structured.free[name] = Rcpp::as<std::vector<double>>(r_list[name]);
                for (const auto& name : param_info.name_fixed) param_info.structured.fixed[name] = Rcpp::as<std::vector<double>>(r_list[name]);
                for (const auto& name : param_info.name_constant) param_info.structured.constant[name] = Rcpp::as<std::vector<double>>(r_list[name]);
            } else {
                ParamGroup cpp_user_params;
                bool is_strc = r_list.containsElementNamed("free") || r_list.containsElementNamed("fixed") || r_list.containsElementNamed("constant");
                if (is_strc) {
                    if (r_list.containsElementNamed("free")) r_obj_to_cpp_map(r_list["free"], cpp_user_params.free);
                    if (r_list.containsElementNamed("fixed")) r_obj_to_cpp_map(r_list["fixed"], cpp_user_params.fixed);
                    if (r_list.containsElementNamed("constant")) r_obj_to_cpp_map(r_list["constant"], cpp_user_params.constant);
                } else {
                    r_obj_to_cpp_map(std_params, cpp_user_params.free);
                }
                param_info = ::modify_params(cpp_user_params);
            }
        } else if (Rcpp::is<Rcpp::NumericVector>(std_params)) {
            ParamGroup cpp_user_params;
            r_obj_to_cpp_map(std_params, cpp_user_params.free);
            param_info = ::modify_params(cpp_user_params);
        }
    } else {
        ParamGroup cpp_user_params;
        param_info = ::modify_params(cpp_user_params);
    }

    std::unordered_map<std::string, UserPrior> cpp_user_priors;
    if (user_priors.length() > 0 && user_priors.hasAttribute("names")) {
        Rcpp::CharacterVector names = user_priors.names();
        for (int i = 0; i < user_priors.size(); ++i) {
            std::string p_name = Rcpp::as<std::string>(names[i]);
            Rcpp::List spec = Rcpp::as<Rcpp::List>(user_priors[i]);
            UserPrior up;
            up.type = Rcpp::as<std::string>(spec["type"]);
            Rcpp::CharacterVector arg_names = spec.names();
            for (int j = 0; j < spec.size(); ++j) {
                std::string key = Rcpp::as<std::string>(arg_names[j]);
                if (key != "type") up.args[key] = Rcpp::as<double>(spec[j]);
            }
            cpp_user_priors[p_name] = up;
        }
    }
    CriterionPrior cp = ::modify_prior(cpp_user_priors, param_info);
    
    std::vector<double> cpp_free_params;
    for (const auto& name : param_info.name_free) {
        const auto& vals = param_info.structured.free.at(name);
        for (double v : vals) cpp_free_params.push_back(v);
    }
    
    Eigen::Map<const Eigen::VectorXd> free_params_eigen(cpp_free_params.data(), cpp_free_params.size());
    return cp.evaluate<double>(free_params_eigen);
}
