#include <Rcpp.h>
#include "Cpp/include/estimate_mle.hpp"

// 使用宏定义包住 include，骗过 Rcpp::sourceCpp 的正则检查
// 将底层的 estimate_mle 和 objective_function 实现一并拉取过来编译
#define CORE_IMPL "Cpp/src/estimate_mle.cpp"
#include CORE_IMPL

#define OBJ_IMPL "Cpp/src/objective_function.cpp"
#include OBJ_IMPL

// 使用匿名命名空间，避免与 wrapper_modify_params.cpp 中的辅助函数产生符号冲突
namespace {
    void r_obj_to_cpp_map_mle(SEXP r_obj, std::unordered_map<std::string, std::vector<double>>& cpp_map) {
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

//' Perform Maximum Likelihood Estimation (MLE) for metaSDT models
//'
//' @description
//' Multi-threaded optimization using NLOPT to find the best fitting parameters 
//' for each subject independently.
//' 
//' @param df The dataset as a data.frame.
//' @param colnames A list mapping standard roles (stim, resp, conf, etc.) to your column names.
//' @param params A list of user-defined parameters (free, fixed, constant).
//' @param model_name The name of the model to fit (e.g., "sdt").
//' 
//' @return A data.frame of fitting results for each subject.
//' @import Rcpp
//' @import nloptr
//' @export
// [[Rcpp::export(name = "estimate_mle")]]
Rcpp::DataFrame r_estimate_mle(
    Rcpp::DataFrame df, 
    Rcpp::Nullable<Rcpp::List> colnames = R_NilValue, 
    Rcpp::Nullable<Rcpp::RObject> params = R_NilValue,
    std::string model = "sdt",
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> lower = R_NilValue,
    Rcpp::Nullable<Rcpp::List> upper = R_NilValue
) {
    // 1. 转换 DataFrame
    std::unordered_map<std::string, std::vector<double>> cpp_df;
    Rcpp::CharacterVector df_names = df.names();
    for (int i = 0; i < df.size(); ++i) {
        int type = TYPEOF(df[i]);
        if (type == INTSXP || type == REALSXP || type == LGLSXP) {
            cpp_df[Rcpp::as<std::string>(df_names[i])] = Rcpp::as<std::vector<double>>(df[i]);
        }
    }

    // 2. 转换列名映射
    std::unordered_map<std::string, std::string> cpp_colnames;
    if (colnames.isNotNull()) {
        Rcpp::List col_list(colnames);
        if (col_list.hasAttribute("names")) {
            Rcpp::CharacterVector c_names = col_list.names();
            for (int i = 0; i < col_list.size(); ++i) {
                cpp_colnames[Rcpp::as<std::string>(c_names[i])] = Rcpp::as<std::string>(col_list[i]);
            }
        }
    }

    // 3. 转换模型参数
    ParamGroup user_params;
    if (params.isNotNull() && Rf_length(params) > 0) {
        Rcpp::RObject p_obj(params);
        if (Rcpp::is<Rcpp::List>(p_obj)) {
            Rcpp::List r_list(p_obj);
            if (r_list.containsElementNamed("free"))   r_obj_to_cpp_map_mle(r_list["free"], user_params.free);
            if (r_list.containsElementNamed("fixed"))  r_obj_to_cpp_map_mle(r_list["fixed"], user_params.fixed);
            if (r_list.containsElementNamed("constant")) r_obj_to_cpp_map_mle(r_list["constant"], user_params.constant);
            if (!r_list.containsElementNamed("free") && !r_list.containsElementNamed("fixed") && !r_list.containsElementNamed("constant")) {
                r_obj_to_cpp_map_mle(p_obj, user_params.free);
            }
        }
    }

    // 4. 解析控制参数与边界
    NLoptControl cpp_control;
    // 注入安全的优化器默认参数，防止未传 control 导致 maxeval=0 而瞬间罢工
    cpp_control.algorithm = "LN_BOBYQA";
    cpp_control.xtol_rel = 1e-6;
    cpp_control.maxeval = 10000;
    cpp_control.ftol_rel = 0.0;
    cpp_control.ftol_abs = 0.0;
    cpp_control.xtol_abs = 0.0;
    cpp_control.maxtime = 0.0;
    cpp_control.stopval = 0.0;
    cpp_control.population = 0;
    cpp_control.initial_step = 0.0;
    cpp_control.local_algorithm = "LN_BOBYQA";

    if (control.isNotNull()) {
        Rcpp::List ctrl(control);
        if (ctrl.containsElementNamed("algorithm")) cpp_control.algorithm = Rcpp::as<std::string>(ctrl["algorithm"]);
        if (ctrl.containsElementNamed("xtol_rel")) cpp_control.xtol_rel = Rcpp::as<double>(ctrl["xtol_rel"]);
        if (ctrl.containsElementNamed("maxeval")) cpp_control.maxeval = Rcpp::as<int>(ctrl["maxeval"]);
        if (ctrl.containsElementNamed("ftol_rel")) cpp_control.ftol_rel = Rcpp::as<double>(ctrl["ftol_rel"]);
        if (ctrl.containsElementNamed("ftol_abs")) cpp_control.ftol_abs = Rcpp::as<double>(ctrl["ftol_abs"]);
        if (ctrl.containsElementNamed("xtol_abs")) cpp_control.xtol_abs = Rcpp::as<double>(ctrl["xtol_abs"]);
        if (ctrl.containsElementNamed("maxtime")) cpp_control.maxtime = Rcpp::as<double>(ctrl["maxtime"]);
        if (ctrl.containsElementNamed("stopval")) cpp_control.stopval = Rcpp::as<double>(ctrl["stopval"]);
        if (ctrl.containsElementNamed("population")) cpp_control.population = Rcpp::as<int>(ctrl["population"]);
        if (ctrl.containsElementNamed("initial_step")) cpp_control.initial_step = Rcpp::as<double>(ctrl["initial_step"]);
        if (ctrl.containsElementNamed("local_algorithm")) cpp_control.local_algorithm = Rcpp::as<std::string>(ctrl["local_algorithm"]);
    }

    std::unordered_map<std::string, std::vector<double>> cpp_lower, cpp_upper;
    if (lower.isNotNull()) r_obj_to_cpp_map_mle(lower, cpp_lower);
    if (upper.isNotNull()) r_obj_to_cpp_map_mle(upper, cpp_upper);

    // 5. 调用底层的 C++ 多线程并行 MLE 优化！
    std::vector<SubjectFitResult> cpp_res = ::estimate_mle(
        cpp_df, cpp_colnames, user_params, model, cpp_control, cpp_lower, cpp_upper
    );

    // 6. 彻底展平结果，采用“列导向”组装，直接在 C++ 端生成原生的 R data.frame
    int n_rows = cpp_res.size();
    
    Rcpp::NumericVector v_subid(n_rows);
    Rcpp::NumericVector v_logL(n_rows);
    Rcpp::NumericVector v_aic(n_rows);
    Rcpp::NumericVector v_bic(n_rows);
    Rcpp::IntegerVector v_status(n_rows);
    
    std::vector<std::string> param_names;
    std::unordered_map<std::string, Rcpp::NumericVector> param_cols;
    
    // 6.1 初始化所有的动态参数列
    if (n_rows > 0) {
        for (const auto& kv : cpp_res[0].best_params) {
            if (kv.second.size() == 1) {
                param_names.push_back(kv.first);
                param_cols[kv.first] = Rcpp::NumericVector(n_rows);
            } else {
                for (size_t j = 0; j < kv.second.size(); ++j) {
                    std::string key_name = kv.first + "_" + std::to_string(j + 1);
                    param_names.push_back(key_name);
                    param_cols[key_name] = Rcpp::NumericVector(n_rows);
                }
            }
        }
    }
    
    // 6.2 填充所有列的数据
    for (int i = 0; i < n_rows; ++i) {
        const auto& r = cpp_res[i];
        v_subid[i] = r.subid;
        v_logL[i] = r.logL;
        v_aic[i] = r.aic;
        v_bic[i] = r.bic;
        v_status[i] = r.status;
        
        for (const auto& kv : r.best_params) {
            if (kv.second.size() == 1) {
                param_cols[kv.first][i] = kv.second[0];
            } else {
                for (size_t j = 0; j < kv.second.size(); ++j) {
                    std::string key_name = kv.first + "_" + std::to_string(j + 1);
                    param_cols[key_name][i] = kv.second[j];
                }
            }
        }
    }
    
    // 6.3 组装为底层的 List 并赋予 data.frame 属性
    Rcpp::List df_out;
    df_out["subid"] = v_subid;
    df_out["logL"] = v_logL;
    df_out["aic"] = v_aic;
    df_out["bic"] = v_bic;
    df_out["status"] = v_status;
    
    for (const auto& name : param_names) {
        df_out[name] = param_cols[name];
    }
    
    // 赋予 R 语言 data.frame 的标志性属性
    df_out.attr("class") = "data.frame";
    df_out.attr("row.names") = Rcpp::seq(1, n_rows);
    
    return df_out;
}