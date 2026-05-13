#include <Rcpp.h>
#include "../../Cpp/include/modify_params.hpp"

#define CORE_IMPL "../../Cpp/src/modify_params.cpp"
#include CORE_IMPL
void r_obj_to_cpp_map(SEXP r_obj, std::unordered_map<std::string, std::vector<double>>& cpp_map) {
    if (Rf_isNull(r_obj) || Rf_length(r_obj) == 0) {
    }

    Rcpp::RObject robj(r_obj);

    if (!robj.hasAttribute("names")) {
        return;
    }
    Rcpp::CharacterVector names = robj.attr("names");

    if (names.isNULL() || Rf_length(names) != Rf_length(robj)) {
        return;
    }

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

//' Modify and flatten model parameters (C++ core)
//'
//' @description
//' A robust function to manage model parameters. It takes user-defined parameters,
//' merges them with a set of defaults, resolves any conflicts based on a
//' `free > fixed > constant` priority, and returns a single flattened Rcpp::List.
//'
//' @param params A Rcpp::List or a Rcpp::Named vector specifying user parameters.
//'   - If `params` is a Rcpp::List containing `free`, `fixed`, or `constant` slots,
//'     it's treated as a structured parameter set.
//'   - If `params` is a simple Rcpp::Named Rcpp::List (e.g., `Rcpp::List(d=2)`) or a Rcpp::Named vector
//'     (e.g., `c(d=2)`), all its elements are treated as **free** parameters by default.
//'   - If `params` is empty or `NULL`, the function returns the default flattened parameter set.
//'
//' @return A Rcpp::Named Rcpp::List containing the final, flattened parameters.
//' @import Rcpp
//' @export
// [[Rcpp::export(name = "modify_params")]]
Rcpp::List r_modify_params(Rcpp::RObject user_params = R_NilValue) {
    ParamGroup cpp_user_params;

    if (!user_params.isNULL() && Rf_length(user_params) > 0) {
        if (Rcpp::is<Rcpp::List>(user_params)) {
            Rcpp::List r_list(user_params);

            bool is_structured = r_list.containsElementNamed("free") || 
                                 r_list.containsElementNamed("fixed") || 
                                 r_list.containsElementNamed("constant");

            if (is_structured) { // Case 1: Structured Rcpp::List like Rcpp::List(free=...)
                if (r_list.containsElementNamed("free"))   
                    r_obj_to_cpp_map(r_list["free"], cpp_user_params.free);
                if (r_list.containsElementNamed("fixed"))  
                    r_obj_to_cpp_map(r_list["fixed"], cpp_user_params.fixed);
                if (r_list.containsElementNamed("constant")) 
                    r_obj_to_cpp_map(r_list["constant"], cpp_user_params.constant);
            } else { // Case 2: Flat Rcpp::List like Rcpp::List(d=2), treat as free
                r_obj_to_cpp_map(user_params, cpp_user_params.free);
            }
        } else if (Rcpp::is<Rcpp::NumericVector>(user_params)) { // Case 3: Flat vector like c(d=2), treat as free
            r_obj_to_cpp_map(user_params, cpp_user_params.free);
        } else {
            Rcpp::stop("Input 'params' must be a Rcpp::List, a Rcpp::Named numeric vector, or NULL.");
        }
    }

    auto cpp_result = ::modify_params(cpp_user_params);

    Rcpp::List out_list = Rcpp::wrap(cpp_result.flat);
    
    out_list["name_free"] = cpp_result.name_free;
    out_list["name_fixed"] = cpp_result.name_fixed;
    out_list["name_constant"] = cpp_result.name_constant;
    
    out_list["lower_bounds"] = cpp_result.lower_bounds;
    out_list["upper_bounds"] = cpp_result.upper_bounds;
    
    out_list["numb_free"] = cpp_result.numb_free;
    out_list["numb_fixed"] = cpp_result.numb_fixed;
    out_list["numb_constant"] = cpp_result.numb_constant;
    
    return out_list;
}
