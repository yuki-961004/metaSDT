#include <Rcpp.h>

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../Cpp/include/estimate_mcmc.hpp"
#include "../../Cpp/include/modify_control.hpp"
#include "../../Cpp/include/modify_outputs.hpp"
#include "../../Cpp/include/modify_prior.hpp"
#include "../../Cpp/include/progress_bar.hpp"

#define R_MODIFY_OUT_IMPL "r_modify_outputs.cpp"
#include R_MODIFY_OUT_IMPL
#define R_CTRL_WRAP_IMPL "r_modify_control.cpp"
#include R_CTRL_WRAP_IMPL

#define ALGO_STAN_IMPL "../../Cpp/src/algorithm_stan.cpp"
#include ALGO_STAN_IMPL
#define MODIFY_OUTPUTS_IMPL "../../Cpp/src/modify_outputs.cpp"
#include MODIFY_OUTPUTS_IMPL
#define CORE_IMPL_MCMC "../../Cpp/src/estimate_mcmc.cpp"
#include CORE_IMPL_MCMC

namespace {

/* ========================================================================== *
 *                            Input Converters                               *
 * ========================================================================== */

void r_obj_to_cpp_map(
    SEXP r_obj,
    std::unordered_map<std::string, std::vector<double>>& cpp_map
) {
    if (Rf_isNull(r_obj) || Rf_length(r_obj) == 0) {
        return;
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
            cpp_map[Rcpp::as<std::string>(names[i])] =
                Rcpp::as<std::vector<double>>(r_list[i]);
        }
    } else if (Rcpp::is<Rcpp::NumericVector>(robj)) {
        Rcpp::NumericVector r_vec(robj);
        for (int i = 0; i < r_vec.size(); ++i) {
            cpp_map[Rcpp::as<std::string>(names[i])] = {r_vec[i]};
        }
    }
}

void r_obj_to_user_priors(
    SEXP r_obj,
    std::unordered_map<std::string, UserPrior>& cpp_priors
) {
    if (Rf_isNull(r_obj) || Rf_length(r_obj) == 0) {
        return;
    }

    Rcpp::RObject robj(r_obj);
    if (!robj.hasAttribute("names") || !Rcpp::is<Rcpp::List>(robj)) {
        return;
    }

    Rcpp::List r_list(robj);
    Rcpp::CharacterVector names = r_list.names();
    for (int i = 0; i < r_list.size(); ++i) {
        const std::string param_name = Rcpp::as<std::string>(names[i]);
        Rcpp::List prior_list(r_list[i]);
        Rcpp::CharacterVector prior_names = prior_list.names();

        UserPrior prior;
        prior.type = "none";

        // 每个 prior list 中 type 是分布名, 其余字段是分布参数
        for (int j = 0; j < prior_list.size(); ++j) {
            const std::string key = Rcpp::as<std::string>(prior_names[j]);
            if (key == "type") {
                prior.type = Rcpp::as<std::string>(prior_list[j]);
            } else {
                prior.args[key] = Rcpp::as<double>(prior_list[j]);
            }
        }

        cpp_priors[param_name] = prior;
    }
}

/* ========================================================================== *
 *                            Output Converters                              *
 * ========================================================================== */

void set_data_frame_attrs(Rcpp::List& out, int n_rows) {
    out.attr("class") = "data.frame";
    out.attr("row.names") = Rcpp::seq(1, n_rows);
}

Rcpp::NumericMatrix draws_to_matrix(
    const std::vector<std::vector<double>>& draws
) {
    if (draws.empty()) {
        return Rcpp::NumericMatrix(0, 0);
    }

    const int n_rows = static_cast<int>(draws.size());
    const int n_cols = static_cast<int>(draws[0].size());
    Rcpp::NumericMatrix out(n_rows, n_cols);

    // 每一行是一次 posterior draw, 每一列是该参数的一个元素
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            out(i, j) = draws[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }

    return out;
}

Rcpp::List samples_to_list(const OutputDiagnosticsRow& row) {
    Rcpp::List out;

    for (const auto& kv : row.samples) {
        out[kv.first] = draws_to_matrix(kv.second);
    }

    out["logPost"] = row.draw_logPost;
    return out;
}

std::string subject_key(const OutputDiagnosticsRow& row) {
    std::string key = "subject_" + std::to_string(row.subid);
    if (!row.cond.empty()) {
        key += "_" + row.cond;
    }
    return key;
}

Rcpp::List create_fit_df(
    const std::vector<OutputFitRow>& rows,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    const int n_rows = static_cast<int>(rows.size());
    Rcpp::NumericVector subid(n_rows);
    Rcpp::NumericVector logL(n_rows);
    Rcpp::NumericVector logPrior(n_rows);
    Rcpp::NumericVector logPost(n_rows);
    Rcpp::NumericVector aic(n_rows);
    Rcpp::NumericVector bic(n_rows);
    Rcpp::IntegerVector status(n_rows);

    std::vector<std::string> param_names;
    std::vector<std::string> sd_names;
    std::unordered_map<std::string, Rcpp::NumericVector> param_cols;
    std::unordered_map<std::string, Rcpp::NumericVector> sd_cols;

    if (!rows.empty()) {
        param_names = r_modify_outputs::ordered_flat_names(
            ordered_params,
            param_sizes,
            rows[0].params
        );
        sd_names = r_modify_outputs::ordered_flat_names(
            ordered_params,
            param_sizes,
            rows[0].param_sd
        );

        for (const auto& name : param_names) {
            param_cols[name] = Rcpp::NumericVector(n_rows, NA_REAL);
        }
        for (const auto& name : sd_names) {
            sd_cols[name + "_sd"] = Rcpp::NumericVector(n_rows, NA_REAL);
        }
    }

    for (int i = 0; i < n_rows; ++i) {
        const OutputFitRow& row = rows[static_cast<size_t>(i)];
        subid[i] = row.subid;
        logL[i] = row.logL;
        logPrior[i] = row.logPrior;
        logPost[i] = row.logPost;
        aic[i] = row.aic;
        bic[i] = row.bic;
        status[i] = row.status;

        for (const auto& kv : row.params) {
            std::vector<double> values = kv.second;
            if (kv.first == "c_conf" && !values.empty()) {
                std::sort(values.begin(), values.end());
            }
            if (values.size() == 1) {
                param_cols[kv.first][i] = values[0];
            } else {
                for (size_t j = 0; j < values.size(); ++j) {
                    const std::string key =
                        kv.first + "_" + std::to_string(j + 1);
                    param_cols[key][i] = values[j];
                }
            }
        }

        for (const auto& kv : row.param_sd) {
            const std::vector<double>& values = kv.second;
            if (values.size() == 1) {
                sd_cols[kv.first + "_sd"][i] = values[0];
            } else {
                for (size_t j = 0; j < values.size(); ++j) {
                    const std::string key =
                        kv.first + "_" + std::to_string(j + 1) + "_sd";
                    sd_cols[key][i] = values[j];
                }
            }
        }
    }

    Rcpp::List out;
    out["subid"] = subid;
    out["logL"] = logL;
    out["logPrior"] = logPrior;
    out["logPost"] = logPost;
    out["aic"] = aic;
    out["bic"] = bic;
    out["status"] = status;

    for (const auto& name : param_names) {
        out[name] = param_cols[name];
    }
    for (const auto& name : sd_names) {
        out[name + "_sd"] = sd_cols[name + "_sd"];
    }

    set_data_frame_attrs(out, n_rows);
    return out;
}

Rcpp::RObject create_fit_slot(
    const OutputFitSlot& slot,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    if (slot.by_condition.size() == 1 &&
        slot.by_condition.begin()->first.empty()) {
        return create_fit_df(
            slot.by_condition.begin()->second,
            ordered_params,
            param_sizes
        );
    }

    Rcpp::List out;
    for (const auto& kv : slot.by_condition) {
        out[kv.first] = create_fit_df(kv.second, ordered_params, param_sizes);
    }
    return out;
}

Rcpp::List create_subject_diag_df(
    const std::vector<OutputDiagnosticsRow>& rows
) {
    const int n_rows = static_cast<int>(rows.size());
    Rcpp::NumericVector subid(n_rows);
    Rcpp::CharacterVector condition(n_rows);
    Rcpp::IntegerVector status(n_rows);
    Rcpp::IntegerVector n_evals(n_rows);
    Rcpp::IntegerVector result_code(n_rows);
    Rcpp::IntegerVector n_draws(n_rows);
    Rcpp::IntegerVector n_chains(n_rows);
    Rcpp::IntegerVector warmup(n_rows);
    Rcpp::IntegerVector thin(n_rows);
    Rcpp::IntegerVector leapfrog_steps(n_rows);
    Rcpp::NumericVector accept_rate(n_rows);
    Rcpp::NumericVector step_size(n_rows);
    Rcpp::CharacterVector result_message(n_rows);
    Rcpp::CharacterVector stop_reason(n_rows);

    for (int i = 0; i < n_rows; ++i) {
        const OutputDiagnosticsRow& row = rows[static_cast<size_t>(i)];
        subid[i] = row.subid;
        condition[i] = row.cond;
        status[i] = row.status;
        n_evals[i] = row.n_evals;
        result_code[i] = row.result_code;
        n_draws[i] = row.n_draws;
        n_chains[i] = row.n_chains;
        warmup[i] = row.warmup;
        thin[i] = row.thin;
        leapfrog_steps[i] = row.leapfrog_steps;
        accept_rate[i] = row.accept_rate;
        step_size[i] = row.step_size;
        result_message[i] = row.result_message;
        stop_reason[i] = row.stop_reason;
    }

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("subid") = subid,
        Rcpp::Named("condition") = condition,
        Rcpp::Named("status") = status,
        Rcpp::Named("n_evals") = n_evals,
        Rcpp::Named("result_code") = result_code,
        Rcpp::Named("n_draws") = n_draws,
        Rcpp::Named("n_chains") = n_chains,
        Rcpp::Named("warmup") = warmup,
        Rcpp::Named("thin") = thin,
        Rcpp::Named("leapfrog_steps") = leapfrog_steps,
        Rcpp::Named("accept_rate") = accept_rate,
        Rcpp::Named("step_size") = step_size,
        Rcpp::Named("result_message") = result_message,
        Rcpp::Named("stop_reason") = stop_reason
    );
    set_data_frame_attrs(out, n_rows);
    return out;
}

Rcpp::List create_condition_diagnostics(
    const std::vector<OutputDiagnosticsRow>& rows
) {
    Rcpp::List posterior_samples;

    // samples 体积可能较大, 因此放 diagnostics 内的独立子槽
    for (const OutputDiagnosticsRow& row : rows) {
        posterior_samples[subject_key(row)] = samples_to_list(row);
    }

    return Rcpp::List::create(
        Rcpp::Named("subject_sampling") = create_subject_diag_df(rows),
        Rcpp::Named("posterior_samples") = posterior_samples
    );
}

Rcpp::List progress_to_list() {
    const ui::ProgressSnapshot ps = ui::progress_last_snapshot();
    return Rcpp::List::create(
        Rcpp::Named("requested_mode") = ps.requested_mode,
        Rcpp::Named("resolved_mode") = ps.resolved_mode,
        Rcpp::Named("elapsed_sec") = ps.elapsed_sec,
        Rcpp::Named("total_iterations") = static_cast<double>(ps.total),
        Rcpp::Named("completed_iterations") =
            static_cast<double>(ps.completed),
        Rcpp::Named("speed") = ps.speed,
        Rcpp::Named("finished") = ps.finished
    );
}

Rcpp::List create_diagnostics_slot(const OutputDiagnosticsSlot& slot) {
    Rcpp::List out;

    if (slot.by_condition.size() == 1 &&
        slot.by_condition.begin()->first.empty()) {
        out = create_condition_diagnostics(
            slot.by_condition.begin()->second
        );
        out["progress"] = progress_to_list();
        return out;
    }

    for (const auto& kv : slot.by_condition) {
        out[kv.first] = create_condition_diagnostics(kv.second);
    }
    out["__progress__"] = progress_to_list();
    return out;
}

} // namespace

// [[Rcpp::export(name = "estimate_mcmc")]]
Rcpp::RObject r_estimate_mcmc(
    Rcpp::DataFrame df,
    Rcpp::Nullable<Rcpp::List> colnames = R_NilValue,
    Rcpp::Nullable<Rcpp::RObject> params = R_NilValue,
    std::string model = "sdt",
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> lower = R_NilValue,
    Rcpp::Nullable<Rcpp::List> upper = R_NilValue,
    Rcpp::Nullable<Rcpp::List> priors = R_NilValue
) {
    std::unordered_map<std::string, std::vector<double>> cpp_df;
    Rcpp::CharacterVector df_names = df.names();

    // 仅数值和逻辑列传入 C++ core, 与 MLE/MAP wrapper 保持一致
    for (int i = 0; i < df.size(); ++i) {
        const int type = TYPEOF(df[i]);
        if (type == INTSXP || type == REALSXP || type == LGLSXP) {
            cpp_df[Rcpp::as<std::string>(df_names[i])] =
                Rcpp::as<std::vector<double>>(df[i]);
        }
    }

    std::unordered_map<std::string, std::string> cpp_colnames;
    if (colnames.isNotNull()) {
        Rcpp::List col_list(colnames);
        if (col_list.hasAttribute("names")) {
            Rcpp::CharacterVector c_names = col_list.names();
            for (int i = 0; i < col_list.size(); ++i) {
                cpp_colnames[Rcpp::as<std::string>(c_names[i])] =
                    Rcpp::as<std::string>(col_list[i]);
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
            if (r_list.containsElementNamed("free")) {
                r_obj_to_cpp_map(r_list["free"], user_params.free);
                r_modify_outputs::collect_param_order(
                    r_list["free"],
                    ordered_params,
                    param_sizes
                );
            }
            if (r_list.containsElementNamed("fixed")) {
                r_obj_to_cpp_map(r_list["fixed"], user_params.fixed);
                r_modify_outputs::collect_param_order(
                    r_list["fixed"],
                    ordered_params,
                    param_sizes
                );
            }
            if (r_list.containsElementNamed("constant")) {
                r_obj_to_cpp_map(r_list["constant"], user_params.constant);
                r_modify_outputs::collect_param_order(
                    r_list["constant"],
                    ordered_params,
                    param_sizes
                );
            }
            if (!r_list.containsElementNamed("free") &&
                !r_list.containsElementNamed("fixed") &&
                !r_list.containsElementNamed("constant")) {
                r_obj_to_cpp_map(p_obj, user_params.free);
                r_modify_outputs::collect_param_order(
                    p_obj,
                    ordered_params,
                    param_sizes
                );
            }
        }
    }

    StanControl cpp_control;
    if (control.isNotNull()) {
        Rcpp::List ctrl(control);
        r_wrapper_modify_control::apply_control_from_list(
            ctrl,
            cpp_control
        );
    }
    cpp_control = modify_control(cpp_control, "mcmc");

    std::unordered_map<std::string, std::vector<double>> cpp_lower;
    std::unordered_map<std::string, std::vector<double>> cpp_upper;
    if (lower.isNotNull()) {
        r_obj_to_cpp_map(lower, cpp_lower);
    }
    if (upper.isNotNull()) {
        r_obj_to_cpp_map(upper, cpp_upper);
    }

    std::unordered_map<std::string, UserPrior> cpp_priors;
    if (priors.isNotNull()) {
        r_obj_to_user_priors(priors, cpp_priors);
    }

    const std::vector<SubjectMCMCResult> cpp_res = ::estimate_mcmc(
        cpp_df,
        cpp_colnames,
        user_params,
        model,
        cpp_control,
        cpp_lower,
        cpp_upper,
        cpp_priors
    );

    const OutputFitSlot fit_slot = modify_outputs::Stan_fit(cpp_res);
    const OutputEstimatorSlot estimator_slot =
        modify_outputs::Stan_estimator("MCMC", cpp_control);
    const OutputDiagnosticsSlot diagnostics_slot =
        modify_outputs::Stan_diagnostics(cpp_res);
    const EstimateOutput output = modify_outputs::combine_output_slots(
        fit_slot,
        estimator_slot,
        diagnostics_slot
    );

    Rcpp::List estimator = Rcpp::List::create(
        Rcpp::Named("name") = estimator_slot.name,
        Rcpp::Named("backend") = estimator_slot.backend,
        Rcpp::Named("algorithm") = estimator_slot.algorithm,
        Rcpp::Named("control") =
            r_wrapper_modify_control::control_to_list(cpp_control)
    );

    return Rcpp::List::create(
        Rcpp::Named("fit") = create_fit_slot(
            output.fit,
            ordered_params,
            param_sizes
        ),
        Rcpp::Named("estimator") = estimator,
        Rcpp::Named("diagnostics") = create_diagnostics_slot(
            output.diagnostics
        )
    );
}
