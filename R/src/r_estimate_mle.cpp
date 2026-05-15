#include <Rcpp.h>
#include <algorithm>
#include "../../Cpp/include/estimate_mle.hpp"
#include "../../Cpp/include/modify_control.hpp"
#define R_MODIFY_OUT_IMPL "r_modify_outputs.cpp"
#include R_MODIFY_OUT_IMPL
#define R_CTRL_WRAP_IMPL "r_modify_control.cpp"
#include R_CTRL_WRAP_IMPL

#define CORE_IMPL "../../Cpp/src/estimate_mle.cpp"
#include CORE_IMPL
#define OBJ_IMPL "../../Cpp/src/task_builder.cpp"
#include OBJ_IMPL
#define CTRL_IMPL "../../Cpp/src/modify_control.cpp"
#include CTRL_IMPL
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
            cpp_map[Rcpp::as<std::string>(names[i])] = {r_vec[i]};
        }
    }
}

}

// [[Rcpp::export(name = "estimate_mle")]]
Rcpp::RObject r_estimate_mle(
    Rcpp::DataFrame df,
    Rcpp::Nullable<Rcpp::List> colnames = R_NilValue,
    Rcpp::Nullable<Rcpp::RObject> params = R_NilValue,
    std::string model = "sdt",
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> lower = R_NilValue,
    Rcpp::Nullable<Rcpp::List> upper = R_NilValue
) {
    std::unordered_map<std::string, std::vector<double>> cpp_df;
    Rcpp::CharacterVector df_names = df.names();
    for (int i = 0; i < df.size(); ++i) {
        int type = TYPEOF(df[i]);
        if (type == INTSXP || type == REALSXP || type == LGLSXP) {
            cpp_df[Rcpp::as<std::string>(df_names[i])] = Rcpp::as<std::vector<double>>(df[i]);
        }
    }

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

    ParamGroup user_params;
    std::vector<std::string> ordered_params;
    std::unordered_map<std::string, size_t> param_sizes;
    if (params.isNotNull() && Rf_length(params) > 0) {
        Rcpp::RObject p_obj(params);
        if (Rcpp::is<Rcpp::List>(p_obj)) {
            Rcpp::List r_list(p_obj);
            if (r_list.containsElementNamed("free")) { r_obj_to_cpp_map(r_list["free"], user_params.free); r_modify_outputs::collect_param_order(r_list["free"], ordered_params, param_sizes); }
            if (r_list.containsElementNamed("fixed")) { r_obj_to_cpp_map(r_list["fixed"], user_params.fixed); r_modify_outputs::collect_param_order(r_list["fixed"], ordered_params, param_sizes); }
            if (r_list.containsElementNamed("constant")) { r_obj_to_cpp_map(r_list["constant"], user_params.constant); r_modify_outputs::collect_param_order(r_list["constant"], ordered_params, param_sizes); }
            if (!r_list.containsElementNamed("free") && !r_list.containsElementNamed("fixed") && !r_list.containsElementNamed("constant")) {
                r_obj_to_cpp_map(p_obj, user_params.free);
                r_modify_outputs::collect_param_order(p_obj, ordered_params, param_sizes);
            }
        }
    }

    NLoptControl cpp_control;
    if (control.isNotNull()) {
        Rcpp::List ctrl(control);
        r_wrapper_modify_control::apply_control_from_list(
            ctrl, cpp_control, false
        );
    }
    cpp_control = modify_control(cpp_control, "mle");

    std::unordered_map<std::string, std::vector<double>> cpp_lower, cpp_upper;
    if (lower.isNotNull()) r_obj_to_cpp_map(lower, cpp_lower);
    if (upper.isNotNull()) r_obj_to_cpp_map(upper, cpp_upper);

    std::vector<SubjectFitResult> cpp_res = ::estimate_mle(cpp_df, cpp_colnames, user_params, model, cpp_control, cpp_lower, cpp_upper);

    std::map<std::string, std::vector<SubjectFitResult>> grouped_res;
    for (const auto& r : cpp_res) grouped_res[r.cond].push_back(r);

    auto create_df = [&](const std::vector<SubjectFitResult>& res_group) {
        int n_rows = res_group.size();
        Rcpp::NumericVector v_subid(n_rows), v_logL(n_rows), v_aic(n_rows), v_bic(n_rows);
        Rcpp::IntegerVector v_status(n_rows);

        std::vector<std::string> param_names;
        std::unordered_map<std::string, Rcpp::NumericVector> param_cols;
        if (n_rows > 0) {
            param_names = r_modify_outputs::ordered_flat_names(ordered_params, param_sizes, res_group[0].best_params);
            for (const auto& name : param_names) {
                param_cols[name] = Rcpp::NumericVector(n_rows);
            }
        }

        for (int i = 0; i < n_rows; ++i) {
            const auto& r = res_group[i];
            v_subid[i] = r.subid;
            v_logL[i] = r.logL;
            v_aic[i] = r.aic;
            v_bic[i] = r.bic;
            v_status[i] = r.status;
            for (const auto& kv : r.best_params) {
                std::vector<double> values = kv.second;
                if (kv.first == "c_conf" && !values.empty()) std::sort(values.begin(), values.end());
                if (values.size() == 1) {
                    param_cols[kv.first][i] = values[0];
                } else {
                    for (size_t j = 0; j < values.size(); ++j) {
                        param_cols[kv.first + "_" + std::to_string(j + 1)][i] = values[j];
                    }
                }
            }
        }

        Rcpp::List df_out;
        df_out["subid"] = v_subid;
        df_out["logL"] = v_logL;
        df_out["aic"] = v_aic;
        df_out["bic"] = v_bic;
        df_out["status"] = v_status;
        for (const auto& name : param_names) df_out[name] = param_cols[name];
        df_out.attr("class") = "data.frame";
        df_out.attr("row.names") = Rcpp::seq(1, n_rows);
        return df_out;
    };

    Rcpp::RObject fit;
    if (grouped_res.size() == 1 && grouped_res.begin()->first == "") {
        fit = create_df(grouped_res.begin()->second);
    } else {
        Rcpp::List out_list;
        for (const auto& kv : grouped_res) out_list[kv.first] = create_df(kv.second);
        fit = out_list;
    }

    Rcpp::List estimator = Rcpp::List::create(
        Rcpp::Named("name") = "MLE",
        Rcpp::Named("backend") = "nlopt",
        Rcpp::Named("global_algorithm") = cpp_control.algorithm,
        Rcpp::Named("local_algorithm") = cpp_control.local_algorithm,
        Rcpp::Named("control") =
            r_wrapper_modify_control::control_to_list(cpp_control, false)
    );

    const int n = static_cast<int>(cpp_res.size());
    Rcpp::NumericVector d_subid(n);
    Rcpp::CharacterVector d_cond(n);
    Rcpp::IntegerVector d_status(n), d_n_evals(n), d_result_code(n);
    Rcpp::NumericVector d_optimum_value(n);
    Rcpp::CharacterVector d_result_message(n), d_stop_reason(n);
    for (int i = 0; i < n; ++i) {
        d_subid[i] = cpp_res[i].subid;
        d_cond[i] = cpp_res[i].cond;
        d_status[i] = cpp_res[i].status;
        d_n_evals[i] = cpp_res[i].n_evals;
        d_result_code[i] = cpp_res[i].status;
        d_optimum_value[i] = cpp_res[i].optimum_value;
        d_result_message[i] = cpp_res[i].result_message;
        d_stop_reason[i] = cpp_res[i].stop_reason;
    }
    Rcpp::List subject_diag = Rcpp::List::create(
        Rcpp::Named("subid") = d_subid,
        Rcpp::Named("condition") = d_cond,
        Rcpp::Named("status") = d_status,
        Rcpp::Named("n_evals") = d_n_evals,
        Rcpp::Named("result_code") = d_result_code,
        Rcpp::Named("result_message") = d_result_message,
        Rcpp::Named("stop_reason") = d_stop_reason,
        Rcpp::Named("optimum_value") = d_optimum_value
    );
    subject_diag.attr("class") = "data.frame";
    subject_diag.attr("row.names") = Rcpp::seq(1, n);

    ui::ProgressSnapshot ps = ui::progress_last_snapshot();
    Rcpp::List diagnostics = Rcpp::List::create(
        Rcpp::Named("subject_optimization") = subject_diag,
        Rcpp::Named("progress") = Rcpp::List::create(
            Rcpp::Named("requested_mode") = ps.requested_mode,
            Rcpp::Named("resolved_mode") = ps.resolved_mode,
            Rcpp::Named("elapsed_sec") = ps.elapsed_sec,
            Rcpp::Named("total_iterations") = static_cast<double>(ps.total),
            Rcpp::Named("completed_iterations") = static_cast<double>(ps.completed),
            Rcpp::Named("speed") = ps.speed,
            Rcpp::Named("finished") = ps.finished
        )
    );

    return Rcpp::List::create(
        Rcpp::Named("fit") = fit,
        Rcpp::Named("estimator") = estimator,
        Rcpp::Named("diagnostics") = diagnostics
    );
}
