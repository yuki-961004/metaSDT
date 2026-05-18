#include <Rcpp.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <abcpp/abcpp_impl.hpp>

#include "../../Cpp/include/estimate_abc.hpp"

#define CORE_IMPL_ABC "../../Cpp/src/estimate_abc.cpp"
#include CORE_IMPL_ABC

namespace {

void set_data_frame_attrs(Rcpp::List& out, int n_rows) {
    out.attr("class") = "data.frame";
    out.attr("row.names") = Rcpp::seq(1, n_rows);
}

std::unordered_map<std::string, std::vector<double>> dataframe_to_cpp(
    Rcpp::DataFrame df
) {
    std::unordered_map<std::string, std::vector<double>> out;
    Rcpp::CharacterVector names = df.names();
    for (int i = 0; i < df.size(); ++i) {
        const int type = TYPEOF(df[i]);
        if (type == INTSXP || type == REALSXP || type == LGLSXP) {
            out[Rcpp::as<std::string>(names[i])] =
                Rcpp::as<std::vector<double>>(df[i]);
        }
    }
    return out;
}

std::unordered_map<std::string, std::string> colnames_to_cpp(
    Rcpp::Nullable<Rcpp::List> colnames
) {
    std::unordered_map<std::string, std::string> out;
    if (colnames.isNotNull()) {
        Rcpp::List col_list(colnames);
        if (col_list.hasAttribute("names")) {
            Rcpp::CharacterVector names = col_list.names();
            for (int i = 0; i < col_list.size(); ++i) {
                out[Rcpp::as<std::string>(names[i])] =
                    Rcpp::as<std::string>(col_list[i]);
            }
        }
    }
    return out;
}

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
    Rcpp::List list(robj);
    for (int i = 0; i < list.size(); ++i) {
        cpp_map[Rcpp::as<std::string>(names[i])] =
            Rcpp::as<std::vector<double>>(list[i]);
    }
}

ParamGroup params_to_cpp(Rcpp::Nullable<Rcpp::RObject> params) {
    ParamGroup out;
    if (params.isNull() || Rf_length(params) == 0) {
        return out;
    }

    Rcpp::RObject obj(params);
    if (!Rcpp::is<Rcpp::List>(obj)) {
        return out;
    }

    Rcpp::List list(obj);
    if (list.containsElementNamed("free")) {
        r_obj_to_cpp_map(list["free"], out.free);
    }
    if (list.containsElementNamed("fixed")) {
        r_obj_to_cpp_map(list["fixed"], out.fixed);
    }
    if (list.containsElementNamed("constant")) {
        r_obj_to_cpp_map(list["constant"], out.constant);
    }
    if (!list.containsElementNamed("free") &&
        !list.containsElementNamed("fixed") &&
        !list.containsElementNamed("constant")) {
        r_obj_to_cpp_map(obj, out.free);
    }
    return out;
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

    Rcpp::List list(robj);
    Rcpp::CharacterVector names = list.names();
    for (int i = 0; i < list.size(); ++i) {
        const std::string param_name = Rcpp::as<std::string>(names[i]);
        Rcpp::List spec(list[i]);
        Rcpp::CharacterVector spec_names = spec.names();

        UserPrior prior;
        prior.type = "none";
        for (int j = 0; j < spec.size(); ++j) {
            const std::string key = Rcpp::as<std::string>(spec_names[j]);
            if (key == "type") {
                prior.type = Rcpp::as<std::string>(spec[j]);
            } else {
                prior.args[key] = Rcpp::as<double>(spec[j]);
            }
        }
        cpp_priors[param_name] = prior;
    }
}

ABCControl control_from_list(Rcpp::Nullable<Rcpp::List> control) {
    ABCControl out;
    if (control.isNull()) {
        return out;
    }

    Rcpp::List ctrl(control);
    if (ctrl.containsElementNamed("tol")) {
        out.tol = Rcpp::as<double>(ctrl["tol"]);
    }
    if (ctrl.containsElementNamed("method")) {
        out.method = Rcpp::as<std::string>(ctrl["method"]);
    }
    if (ctrl.containsElementNamed("reduction")) {
        out.reduction = Rcpp::as<std::string>(ctrl["reduction"]);
    }
    if (ctrl.containsElementNamed("n_comp")) {
        out.n_comp = Rcpp::as<int>(ctrl["n_comp"]);
    }
    if (ctrl.containsElementNamed("samples")) {
        out.samples = Rcpp::as<int>(ctrl["samples"]);
    }
    if (ctrl.containsElementNamed("kernel")) {
        out.kernel = Rcpp::as<std::string>(ctrl["kernel"]);
    }
    if (ctrl.containsElementNamed("hcorr")) {
        out.hcorr = Rcpp::as<bool>(ctrl["hcorr"]);
    }
    if (ctrl.containsElementNamed("seed")) {
        out.seed = Rcpp::as<unsigned int>(ctrl["seed"]);
    }
    if (ctrl.containsElementNamed("print_level")) {
        out.print_level = Rcpp::as<int>(ctrl["print_level"]);
    }
    return out;
}

Rcpp::List control_to_list(const ABCControl& control) {
    return Rcpp::List::create(
        Rcpp::Named("tol") = control.tol,
        Rcpp::Named("method") = control.method,
        Rcpp::Named("reduction") = control.reduction,
        Rcpp::Named("n_comp") = control.n_comp,
        Rcpp::Named("samples") = control.samples,
        Rcpp::Named("kernel") = control.kernel,
        Rcpp::Named("hcorr") = control.hcorr,
        Rcpp::Named("seed") = control.seed,
        Rcpp::Named("print_level") = control.print_level
    );
}

int estimator_n_comp(
    const ABCControl& control,
    const std::vector<SubjectABCResult>& results
) {
    if (control.n_comp > 0) {
        return control.n_comp;
    }
    for (const auto& result : results) {
        if (result.n_comp_used > 0) {
            return result.n_comp_used;
        }
    }
    return 0;
}

Rcpp::List create_fit_df(const std::vector<SubjectABCResult>& rows) {
    const int n_rows = static_cast<int>(rows.size());
    Rcpp::NumericVector subid(n_rows);
    Rcpp::IntegerVector status(n_rows);

    std::vector<std::string> param_names;
    std::unordered_map<std::string, Rcpp::NumericVector> param_cols;
    if (!rows.empty()) {
        param_names = rows[0].parameter_names;
        for (const auto& name : param_names) {
            param_cols[name] = Rcpp::NumericVector(n_rows, NA_REAL);
        }
    }

    for (int i = 0; i < n_rows; ++i) {
        const SubjectABCResult& row = rows[static_cast<std::size_t>(i)];
        subid[i] = row.subid;
        status[i] = row.status;
        const std::size_t n = std::min(
            row.parameter_names.size(),
            row.summary.size()
        );
        for (std::size_t j = 0; j < n; ++j) {
            param_cols[row.parameter_names[j]][i] = row.summary[j].mean;
        }
    }

    Rcpp::List out;
    out["subid"] = subid;
    out["status"] = status;
    for (const auto& name : param_names) {
        out[name] = param_cols[name];
    }
    set_data_frame_attrs(out, n_rows);
    return out;
}

Rcpp::List create_summary_df(const SubjectABCResult& result) {
    const int n_rows = static_cast<int>(
        std::min(result.parameter_names.size(), result.summary.size())
    );
    Rcpp::CharacterVector parameter(n_rows);
    Rcpp::NumericVector min(n_rows), q_lower(n_rows), median(n_rows);
    Rcpp::NumericVector mean(n_rows), mode(n_rows), q_upper(n_rows);
    Rcpp::NumericVector max(n_rows), sd(n_rows);

    for (int i = 0; i < n_rows; ++i) {
        const ABCSummaryStats& s =
            result.summary[static_cast<std::size_t>(i)];
        parameter[i] = result.parameter_names[static_cast<std::size_t>(i)];
        min[i] = s.min;
        q_lower[i] = s.q_lower;
        median[i] = s.median;
        mean[i] = s.mean;
        mode[i] = s.mode;
        q_upper[i] = s.q_upper;
        max[i] = s.max;
        sd[i] = s.sd;
    }

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("parameter") = parameter,
        Rcpp::Named("min") = min,
        Rcpp::Named("q_lower") = q_lower,
        Rcpp::Named("median") = median,
        Rcpp::Named("mean") = mean,
        Rcpp::Named("mode") = mode,
        Rcpp::Named("q_upper") = q_upper,
        Rcpp::Named("max") = max,
        Rcpp::Named("sd") = sd
    );
    set_data_frame_attrs(out, n_rows);
    return out;
}

std::string subject_key(const SubjectABCResult& result) {
    std::string key = "subject_" + std::to_string(result.subid);
    if (!result.cond.empty()) {
        key += "_" + result.cond;
    }
    return key;
}

Rcpp::List create_subject_abc_df(
    const std::vector<SubjectABCResult>& rows
) {
    const int n_rows = static_cast<int>(rows.size());
    Rcpp::NumericVector subid(n_rows);
    Rcpp::CharacterVector condition(n_rows), message(n_rows);
    Rcpp::IntegerVector status(n_rows), n_accepted(n_rows);
    Rcpp::IntegerVector n_comp(n_rows);

    for (int i = 0; i < n_rows; ++i) {
        const SubjectABCResult& row = rows[static_cast<std::size_t>(i)];
        subid[i] = row.subid;
        condition[i] = row.cond;
        status[i] = row.status;
        message[i] = row.message;
        n_comp[i] = row.n_comp_used;
        n_accepted[i] = static_cast<int>(row.accepted_indices.size());
    }

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("subid") = subid,
        Rcpp::Named("condition") = condition,
        Rcpp::Named("status") = status,
        Rcpp::Named("message") = message,
        Rcpp::Named("n_comp") = n_comp,
        Rcpp::Named("n_accepted") = n_accepted
    );
    set_data_frame_attrs(out, n_rows);
    return out;
}

} // namespace

// [[Rcpp::export(name = "estimate_abc")]]
Rcpp::RObject r_estimate_abc(
    Rcpp::DataFrame df,
    Rcpp::Nullable<Rcpp::List> colnames = R_NilValue,
    Rcpp::Nullable<Rcpp::RObject> params = R_NilValue,
    std::string model = "sdt",
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> priors = R_NilValue
) {
    const auto cpp_df = dataframe_to_cpp(df);
    const auto cpp_colnames = colnames_to_cpp(colnames);
    const ParamGroup user_params = params_to_cpp(params);
    const ABCControl cpp_control = control_from_list(control);
    std::unordered_map<std::string, UserPrior> cpp_priors;
    if (priors.isNotNull()) {
        r_obj_to_user_priors(priors, cpp_priors);
    }

    const std::vector<SubjectABCResult> cpp_res = ::estimate_abc(
        cpp_df,
        cpp_colnames,
        user_params,
        model,
        cpp_control,
        cpp_priors
    );

    std::map<std::string, std::vector<SubjectABCResult>> grouped;
    for (const auto& row : cpp_res) {
        grouped[row.cond].push_back(row);
    }

    Rcpp::RObject fit;
    if (grouped.size() == 1 && grouped.begin()->first.empty()) {
        fit = create_fit_df(grouped.begin()->second);
    } else {
        Rcpp::List fit_by_condition;
        for (const auto& kv : grouped) {
            fit_by_condition[kv.first] = create_fit_df(kv.second);
        }
        fit = fit_by_condition;
    }

    Rcpp::List summary;
    for (const auto& row : cpp_res) {
        summary[subject_key(row)] = create_summary_df(row);
    }

    const int n_comp_used = estimator_n_comp(cpp_control, cpp_res);
    Rcpp::List control_used = control_to_list(cpp_control);
    control_used["n_comp"] = n_comp_used;

    Rcpp::List estimator = Rcpp::List::create(
        Rcpp::Named("name") = "ABC",
        Rcpp::Named("backend") = "abcpp",
        Rcpp::Named("method") = cpp_control.method,
        Rcpp::Named("reduction") = cpp_control.reduction,
        Rcpp::Named("control") = control_used
    );

    Rcpp::List diagnostics = Rcpp::List::create(
        Rcpp::Named("subject_abc") = create_subject_abc_df(cpp_res),
        Rcpp::Named("summary") = summary
    );

    return Rcpp::List::create(
        Rcpp::Named("fit") = fit,
        Rcpp::Named("estimator") = estimator,
        Rcpp::Named("diagnostics") = diagnostics
    );
}
