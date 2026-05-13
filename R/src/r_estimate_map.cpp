#include <Rcpp.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>
#include "../../Cpp/include/estimate_map.hpp"
#include "../../Cpp/include/modify_control.hpp"
#include "../../Cpp/include/modify_prior.hpp"
#define R_MODIFY_OUT_IMPL "r_modify_outputs.cpp"
#include R_MODIFY_OUT_IMPL
#define R_CTRL_WRAP_IMPL "r_modify_control.cpp"
#include R_CTRL_WRAP_IMPL

#define CORE_IMPL_MAP "../../Cpp/src/estimate_map.cpp"
#include CORE_IMPL_MAP

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

void r_obj_to_user_priors(SEXP r_obj, std::unordered_map<std::string, UserPrior>& cpp_priors) {
    if (Rf_isNull(r_obj) || Rf_length(r_obj) == 0) return;
    Rcpp::RObject robj(r_obj);
    if (!robj.hasAttribute("names")) return;
    Rcpp::CharacterVector names = robj.attr("names");

    if (Rcpp::is<Rcpp::List>(robj)) {
        Rcpp::List r_list(robj);
        for (int i = 0; i < r_list.size(); ++i) {
            std::string param_name = Rcpp::as<std::string>(names[i]);
            Rcpp::List prior_list(r_list[i]);
            Rcpp::CharacterVector prior_names = prior_list.attr("names");
            UserPrior up;
            up.type = "none";
            for (int j = 0; j < prior_list.size(); ++j) {
                std::string key = Rcpp::as<std::string>(prior_names[j]);
                if (key == "type") up.type = Rcpp::as<std::string>(prior_list[j]);
                else up.args[key] = Rcpp::as<double>(prior_list[j]);
            }
            cpp_priors[param_name] = up;
        }
    }
}


Rcpp::List summarize_distribution(const std::string& type, const std::vector<double>& vals) {
    if (vals.empty()) {
        return Rcpp::List::create(Rcpp::Named("distribution") = type, Rcpp::Named("parameters") = Rcpp::List::create());
    }

    double mean = 0.0;
    for (double v : vals) mean += v;
    mean /= static_cast<double>(vals.size());

    double var = 0.0;
    if (vals.size() >= 2) {
        for (double v : vals) var += (v - mean) * (v - mean);
        var /= static_cast<double>(vals.size() - 1);
    }
    if (var < 1e-6) var = 1e-6;

    std::string t = type;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (t == "norm") t = "normal";
    if (t == "lnorm") t = "lognormal";
    if (t == "exp") t = "exponential";
    if (t == "unif") t = "uniform";

    if (t == "normal") {
        return Rcpp::List::create(Rcpp::Named("distribution") = "normal", Rcpp::Named("parameters") = Rcpp::List::create(Rcpp::Named("mean") = mean, Rcpp::Named("sd") = std::sqrt(var)));
    }
    if (t == "exponential") {
        return Rcpp::List::create(Rcpp::Named("distribution") = "exponential", Rcpp::Named("parameters") = Rcpp::List::create(Rcpp::Named("rate") = 1.0 / std::max(mean, 1e-6)));
    }
    if (t == "uniform") {
        auto mm = std::minmax_element(vals.begin(), vals.end());
        return Rcpp::List::create(Rcpp::Named("distribution") = "uniform", Rcpp::Named("parameters") = Rcpp::List::create(Rcpp::Named("min") = *mm.first, Rcpp::Named("max") = *mm.second));
    }
    if (t == "lognormal") {
        std::vector<double> lv(vals.size());
        double lmean = 0.0;
        for (size_t i = 0; i < vals.size(); ++i) {
            lv[i] = std::log(std::max(vals[i], 1e-8));
            lmean += lv[i];
        }
        lmean /= static_cast<double>(vals.size());
        double lvar = 0.0;
        if (vals.size() >= 2) {
            for (double v : lv) lvar += (v - lmean) * (v - lmean);
            lvar /= static_cast<double>(vals.size() - 1);
        }
        if (lvar < 1e-6) lvar = 1e-6;
        return Rcpp::List::create(Rcpp::Named("distribution") = "lognormal", Rcpp::Named("parameters") = Rcpp::List::create(Rcpp::Named("meanlog") = lmean, Rcpp::Named("sdlog") = std::sqrt(lvar)));
    }
    if (t == "beta") {
        double m = std::max(1e-4, std::min(mean, 1.0 - 1e-4));
        double max_var = m * (1.0 - m) - 1e-6;
        double v = std::min(var, max_var);
        double common = (m * (1.0 - m) / std::max(v, 1e-8)) - 1.0;
        double shape1 = std::max(m * common, 1e-3);
        double shape2 = std::max((1.0 - m) * common, 1e-3);
        return Rcpp::List::create(Rcpp::Named("distribution") = "beta", Rcpp::Named("parameters") = Rcpp::List::create(Rcpp::Named("shape1") = shape1, Rcpp::Named("shape2") = shape2));
    }
    if (t == "cauchy") {
        std::vector<double> x = vals;
        std::sort(x.begin(), x.end());
        const size_t n = x.size();
        const double location = (n % 2 == 0) ? (x[n / 2 - 1] + x[n / 2]) / 2.0 : x[n / 2];
        const double scale = std::max((x[(3 * n) / 4] - x[n / 4]) / 2.0, 1e-4);
        return Rcpp::List::create(Rcpp::Named("distribution") = "cauchy", Rcpp::Named("parameters") = Rcpp::List::create(Rcpp::Named("location") = location, Rcpp::Named("scale") = scale));
    }

    return Rcpp::List::create(Rcpp::Named("distribution") = "none", Rcpp::Named("parameters") = Rcpp::List::create());
}

Rcpp::List summarize_posteriors(
    const std::vector<SubjectFitResult>& cpp_res,
    const std::unordered_map<std::string, UserPrior>& user_priors,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    if (cpp_res.empty()) return Rcpp::List::create();

    std::unordered_map<std::string, UserPrior> prior_map = default_priors();
    for (const auto& kv : user_priors) prior_map[kv.first] = kv.second;

    std::vector<std::string> base_order = r_modify_outputs::ordered_base_names(ordered_params, cpp_res[0].best_params);

    Rcpp::List out;
    for (const auto& base : base_order) {
        std::string type = "none";
        auto itp = prior_map.find(base);
        if (itp != prior_map.end()) type = itp->second.type;

        size_t p_len = 0;
        auto itsz = param_sizes.find(base);
        if (itsz != param_sizes.end()) p_len = itsz->second;
        auto it_obs = cpp_res[0].best_params.find(base);
        if (it_obs != cpp_res[0].best_params.end()) p_len = std::max(p_len, it_obs->second.size());
        if (p_len == 0) p_len = 1;
        for (size_t j = 0; j < p_len; ++j) {
            std::vector<double> vals;
            vals.reserve(cpp_res.size());
            for (const auto& r : cpp_res) {
                auto it = r.best_params.find(base);
                if (it != r.best_params.end() && j < it->second.size()) vals.push_back(it->second[j]);
            }
            std::string key = (p_len == 1) ? base : (base + "_" + std::to_string(j + 1));
            out[key] = summarize_distribution(type, vals);
        }
    }
    return out;
}

Rcpp::List build_solution(
    const std::vector<SubjectFitResult>& cond_res,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    Rcpp::List out;
    if (cond_res.empty()) return out;

    std::vector<std::string> base_order = r_modify_outputs::ordered_base_names(ordered_params, cond_res[0].best_params);

    for (const auto& base : base_order) {
        size_t p_len = 0;
        auto itsz = param_sizes.find(base);
        if (itsz != param_sizes.end()) p_len = itsz->second;
        auto it_obs = cond_res[0].best_params.find(base);
        if (it_obs != cond_res[0].best_params.end()) p_len = std::max(p_len, it_obs->second.size());
        if (p_len == 0) p_len = 1;

        for (size_t j = 0; j < p_len; ++j) {
            std::vector<double> vals;
            vals.reserve(cond_res.size());
            for (const auto& r : cond_res) {
                auto it = r.best_params.find(base);
                if (it != r.best_params.end() && j < it->second.size()) vals.push_back(it->second[j]);
            }
            std::string key = (p_len == 1) ? base : (base + "_" + std::to_string(j + 1));
            if (vals.empty()) out[key] = R_NilValue;
            else out[key] = vals;
        }
    }
    return out;
}

Rcpp::List build_condition_diagnostics(
    const std::vector<SubjectFitResult>& cond_res,
    const std::unordered_map<std::string, UserPrior>& cpp_priors,
    int em_iterations,
    const std::string& stop_reason,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    const int n = static_cast<int>(cond_res.size());
    Rcpp::NumericVector subid(n);
    Rcpp::IntegerVector status(n), n_evals(n), result_code(n);
    Rcpp::NumericVector optimum_value(n);
    Rcpp::CharacterVector result_message(n), stop_reason_vec(n);
    for (int i = 0; i < n; ++i) {
        subid[i] = cond_res[i].subid;
        status[i] = cond_res[i].status;
        n_evals[i] = cond_res[i].n_evals;
        result_code[i] = cond_res[i].status;
        optimum_value[i] = cond_res[i].optimum_value;
        result_message[i] = cond_res[i].result_message;
        stop_reason_vec[i] = cond_res[i].stop_reason;
    }
    Rcpp::List subject_opt = Rcpp::List::create(
        Rcpp::Named("subid") = subid,
        Rcpp::Named("status") = status,
        Rcpp::Named("n_evals") = n_evals,
        Rcpp::Named("result_code") = result_code,
        Rcpp::Named("result_message") = result_message,
        Rcpp::Named("stop_reason") = stop_reason_vec,
        Rcpp::Named("optimum_value") = optimum_value
    );
    subject_opt.attr("class") = "data.frame";
    subject_opt.attr("row.names") = Rcpp::seq(1, n);

    return Rcpp::List::create(
        Rcpp::Named("em_iterations") = em_iterations,
        Rcpp::Named("em_stop_reason") = stop_reason,
        Rcpp::Named("subject_optimization") = subject_opt,
        Rcpp::Named("posterior") = summarize_posteriors(cond_res, cpp_priors, ordered_params, param_sizes),
        Rcpp::Named("solution") = build_solution(cond_res, ordered_params, param_sizes)
    );
}
}

// [[Rcpp::export(name = "estimate_map")]]
Rcpp::RObject r_estimate_map(
    Rcpp::DataFrame df,
    Rcpp::Nullable<Rcpp::List> colnames = R_NilValue,
    Rcpp::Nullable<Rcpp::RObject> params = R_NilValue,
    std::string model = "sdt",
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> lower = R_NilValue,
    Rcpp::Nullable<Rcpp::List> upper = R_NilValue,
    Rcpp::Nullable<Rcpp::List> user_priors = R_NilValue
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
            ctrl, cpp_control, true
        );
    }
    cpp_control = modify_control(cpp_control, "map");

    std::unordered_map<std::string, std::vector<double>> cpp_lower, cpp_upper;
    if (lower.isNotNull()) r_obj_to_cpp_map(lower, cpp_lower);
    if (upper.isNotNull()) r_obj_to_cpp_map(upper, cpp_upper);

    std::unordered_map<std::string, UserPrior> cpp_priors;
    if (user_priors.isNotNull()) r_obj_to_user_priors(user_priors, cpp_priors);

    int em_iterations_used = 0;
    std::unordered_map<std::string, int> em_by_cond;
    std::unordered_map<std::string, std::string> stop_by_cond;
    std::vector<SubjectFitResult> cpp_res = ::estimate_map(
        cpp_df, cpp_colnames, user_params, model, cpp_control, cpp_lower, cpp_upper, cpp_priors,
        &em_iterations_used, &em_by_cond, &stop_by_cond
    );

    std::map<std::string, std::vector<SubjectFitResult>> grouped_res;
    for (const auto& r : cpp_res) grouped_res[r.cond].push_back(r);

    auto create_df = [&](const std::vector<SubjectFitResult>& res_group) {
        int n_rows = res_group.size();
        Rcpp::NumericVector v_subid(n_rows), v_logL(n_rows), v_logPrior(n_rows), v_logPost(n_rows), v_aic(n_rows), v_bic(n_rows);
        Rcpp::IntegerVector v_status(n_rows);

        std::vector<std::string> param_names;
        std::unordered_map<std::string, Rcpp::NumericVector> param_cols;
        if (n_rows > 0) {
            param_names = r_modify_outputs::ordered_flat_names(ordered_params, param_sizes, res_group[0].best_params);
            for (const auto& name : param_names) param_cols[name] = Rcpp::NumericVector(n_rows);
        }

        for (int i = 0; i < n_rows; ++i) {
            const auto& r = res_group[i];
            v_subid[i] = r.subid;
            v_logL[i] = r.logL;
            v_logPrior[i] = r.logPrior;
            v_logPost[i] = r.logPost;
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
        df_out["logPrior"] = v_logPrior;
        df_out["logPost"] = v_logPost;
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
        Rcpp::Named("name") = "MAP",
        Rcpp::Named("backend") = "nlopt",
        Rcpp::Named("global_algorithm") = cpp_control.algorithm,
        Rcpp::Named("local_algorithm") = cpp_control.local_algorithm,
        Rcpp::Named("control") =
            r_wrapper_modify_control::control_to_list(cpp_control, true)
    );

    Rcpp::List diagnostics;
    if (grouped_res.size() == 1 && grouped_res.begin()->first == "") {
        int em_iter = em_iterations_used;
        std::string reason = "unknown";
        auto it_i = em_by_cond.find("");
        if (it_i != em_by_cond.end()) em_iter = it_i->second;
        auto it_r = stop_by_cond.find("");
        if (it_r != stop_by_cond.end()) reason = it_r->second;
        diagnostics = build_condition_diagnostics(
            grouped_res.begin()->second, cpp_priors, em_iter, reason, ordered_params, param_sizes
        );
    } else {
        for (const auto& kv : grouped_res) {
            int em_iter = 0;
            std::string reason = "unknown";
            auto it_i = em_by_cond.find(kv.first);
            if (it_i != em_by_cond.end()) em_iter = it_i->second;
            auto it_r = stop_by_cond.find(kv.first);
            if (it_r != stop_by_cond.end()) reason = it_r->second;
            diagnostics[kv.first] = build_condition_diagnostics(
                kv.second, cpp_priors, em_iter, reason, ordered_params, param_sizes
            );
        }
    }

    ui::ProgressSnapshot ps = ui::progress_last_snapshot();
    Rcpp::List progress_diag = Rcpp::List::create(
        Rcpp::Named("requested_mode") = ps.requested_mode,
        Rcpp::Named("resolved_mode") = ps.resolved_mode,
        Rcpp::Named("elapsed_sec") = ps.elapsed_sec,
        Rcpp::Named("total_iterations") = static_cast<double>(ps.total),
        Rcpp::Named("completed_iterations") = static_cast<double>(ps.completed),
        Rcpp::Named("speed") = ps.speed,
        Rcpp::Named("finished") = ps.finished
    );
    if (grouped_res.size() == 1 && grouped_res.begin()->first == "") {
        diagnostics["progress"] = progress_diag;
    } else {
        diagnostics["__progress__"] = progress_diag;
    }

    return Rcpp::List::create(
        Rcpp::Named("fit") = fit,
        Rcpp::Named("estimator") = estimator,
        Rcpp::Named("diagnostics") = diagnostics
    );
}
