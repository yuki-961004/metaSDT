#include <Rcpp.h>
#include "Cpp/include/modify_params.hpp"

// 使用宏定义包住 include，骗过 Rcpp::sourceCpp 的正则检查
// 避免它自作主张将 cpp 文件单独抽出编译，从而解决"重复链接"和"未定义引用"的问题
#define CORE_IMPL "Cpp/src/modify_params.cpp"
#include CORE_IMPL

using namespace Rcpp;

// 辅助函数：将 R 端的 list 或 named vector 安全地转换为 C++ 的 map
void r_obj_to_cpp_map(SEXP r_obj, std::unordered_map<std::string, std::vector<double>>& cpp_map) {
    if (Rf_isNull(r_obj) || Rf_length(r_obj) == 0) {
        return; // 如果传入的是 NULL 或空对象，则直接返回
    }

    Rcpp::RObject robj(r_obj);

    if (!robj.hasAttribute("names")) {
        return;
    }
    Rcpp::CharacterVector names = robj.attr("names");

    // 必须是完全命名的对象才能被解析
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
            // R 的命名向量中每个元素都是标量，转为 C++ 这边长度为 1 的 vector
            cpp_map[Rcpp::as<std::string>(names[i])] = { r_vec[i] };
        }
    }
}

//' Modify and flatten model parameters (C++ core)
//'
//' @description
//' A robust function to manage model parameters. It takes user-defined parameters,
//' merges them with a set of defaults, resolves any conflicts based on a
//' `free > fixed > constant` priority, and returns a single flattened list.
//'
//' @param params A list or a named vector specifying user parameters.
//'   - If `params` is a list containing `free`, `fixed`, or `constant` slots,
//'     it's treated as a structured parameter set.
//'   - If `params` is a simple named list (e.g., `list(d=2)`) or a named vector
//'     (e.g., `c(d=2)`), all its elements are treated as **free** parameters by default.
//'   - If `params` is empty or `NULL`, the function returns the default flattened parameter set.
//'
//' @return A named list containing the final, flattened parameters.
//' @import Rcpp
//' @export
// [[Rcpp::export]]
List modify_params(RObject params = R_NilValue) {
    ParamGroup user_params;

    if (!params.isNULL() && Rf_length(params) > 0) {
        if (Rcpp::is<Rcpp::List>(params)) {
            Rcpp::List r_list(params);

            bool is_structured = r_list.containsElementNamed("free") || 
                                 r_list.containsElementNamed("fixed") || 
                                 r_list.containsElementNamed("constant");

            if (is_structured) { // Case 1: Structured list like list(free=...)
                if (r_list.containsElementNamed("free"))   
                    r_obj_to_cpp_map(r_list["free"], user_params.free);
                if (r_list.containsElementNamed("fixed"))  
                    r_obj_to_cpp_map(r_list["fixed"], user_params.fixed);
                if (r_list.containsElementNamed("constant")) 
                    r_obj_to_cpp_map(r_list["constant"], user_params.constant);
            } else { // Case 2: Flat list like list(d=2), treat as free
                r_obj_to_cpp_map(params, user_params.free);
            }
        } else if (Rcpp::is<Rcpp::NumericVector>(params)) { // Case 3: Flat vector like c(d=2), treat as free
            r_obj_to_cpp_map(params, user_params.free);
        } else {
            stop("Input 'params' must be a list, a named numeric vector, or NULL.");
        }
    }

    // 调用纯 C++ 核心函数
    auto cpp_result = ::modify_and_flatten_params(user_params);

    // 将 C++ 的 flat 结果转回 R 的 List 作为主体
    List out_list = Rcpp::wrap(cpp_result.flat);
    
    // 直接将名称与数量信息附加到 R 的 List 中
    out_list["name_free"] = cpp_result.name_free;
    out_list["name_fixed"] = cpp_result.name_fixed;
    out_list["name_constant"] = cpp_result.name_constant;
    
    out_list["numb_free"] = cpp_result.numb_free;
    out_list["numb_fixed"] = cpp_result.numb_fixed;
    out_list["numb_constant"] = cpp_result.numb_constant;
    
    return out_list;
}