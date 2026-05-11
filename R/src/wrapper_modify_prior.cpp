#include <Rcpp.h>
#include "../../Cpp/include/modify_params.hpp"
#include "../../Cpp/include/modify_prior.hpp"

#define CORE_IMPL2 "../../Cpp/src/modify_prior.cpp"
#include CORE_IMPL2

using namespace Rcpp;

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

//' Modify and align prior distributions
//' @export
// [[Rcpp::export(name = "modify_prior")]]
List r_modify_prior(List user_priors, RObject std_params = R_NilValue) {
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
        CharacterVector names = user_priors.names();
        for (int i = 0; i < user_priors.size(); ++i) {
            std::string p_name = as<std::string>(names[i]);
            List spec = as<List>(user_priors[i]);
            UserPrior up;
            up.type = as<std::string>(spec["type"]);
            CharacterVector arg_names = spec.names();
            for (int j = 0; j < spec.size(); ++j) {
                std::string key = as<std::string>(arg_names[j]);
                if (key != "type") up.args[key] = as<double>(spec[j]);
            }
            cpp_user_priors[p_name] = up;
        }
    }

    CriterionPrior cp = ::modify_prior(cpp_user_priors, param_info);

    List out;
    for (const auto& kv : cp.prior_specs_) {
        std::string type_str;
        switch(kv.second.type) {
            case CriterionPrior::PriorType::NORMAL: type_str = "normal"; break;
            case CriterionPrior::PriorType::UNIFORM: type_str = "uniform"; break;
            case CriterionPrior::PriorType::LOGNORMAL: type_str = "lognormal"; break;
            case CriterionPrior::PriorType::CAUCHY: type_str = "cauchy"; break;
            case CriterionPrior::PriorType::BETA: type_str = "beta"; break;
            case CriterionPrior::PriorType::EXPONENTIAL: type_str = "exponential"; break;
            case CriterionPrior::PriorType::NONE: type_str = "none"; break;
        }
        out[std::to_string(kv.first)] = List::create(
            Named("type") = type_str, Named("param1") = kv.second.param1, Named("param2") = kv.second.param2
        );
    }
    return out;
}