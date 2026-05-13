#include <Rcpp.h>
#include "../../Cpp/include/modify_control.hpp"

namespace r_wrapper_modify_control {

inline void apply_control_from_list(
    const Rcpp::List& ctrl,
    NLoptControl& out,
    bool include_em
) {
    if (ctrl.containsElementNamed("algorithm")) {
        out.algorithm = Rcpp::as<std::string>(ctrl["algorithm"]);
    }
    if (ctrl.containsElementNamed("local_algorithm")) {
        out.local_algorithm = Rcpp::as<std::string>(ctrl["local_algorithm"]);
    }
    if (ctrl.containsElementNamed("xtol_rel")) {
        out.xtol_rel = Rcpp::as<double>(ctrl["xtol_rel"]);
    }
    if (ctrl.containsElementNamed("xtol_abs")) {
        out.xtol_abs = Rcpp::as<double>(ctrl["xtol_abs"]);
    }
    if (ctrl.containsElementNamed("ftol_rel")) {
        out.ftol_rel = Rcpp::as<double>(ctrl["ftol_rel"]);
    }
    if (ctrl.containsElementNamed("ftol_abs")) {
        out.ftol_abs = Rcpp::as<double>(ctrl["ftol_abs"]);
    }
    if (ctrl.containsElementNamed("maxeval")) {
        out.maxeval = Rcpp::as<int>(ctrl["maxeval"]);
    }
    if (ctrl.containsElementNamed("maxtime")) {
        out.maxtime = Rcpp::as<double>(ctrl["maxtime"]);
    }
    if (ctrl.containsElementNamed("stopval")) {
        out.stopval = Rcpp::as<double>(ctrl["stopval"]);
    }
    if (ctrl.containsElementNamed("population")) {
        out.population = Rcpp::as<int>(ctrl["population"]);
    }
    if (ctrl.containsElementNamed("initial_step")) {
        out.initial_step = Rcpp::as<double>(ctrl["initial_step"]);
    }
    if (ctrl.containsElementNamed("print_level")) {
        out.print_level = Rcpp::as<int>(ctrl["print_level"]);
    }
    if (ctrl.containsElementNamed("progress") && !Rf_isNull(ctrl["progress"])) {
        out.progress = Rcpp::as<std::string>(ctrl["progress"]);
    }
    if (ctrl.containsElementNamed("progress_refresh_ms")) {
        out.progress_refresh_ms = Rcpp::as<int>(ctrl["progress_refresh_ms"]);
    }
    if (ctrl.containsElementNamed("progress_line_interval_sec")) {
        out.progress_line_interval_sec =
            Rcpp::as<double>(ctrl["progress_line_interval_sec"]);
    }
    if (ctrl.containsElementNamed("progress_line_interval_pct")) {
        out.progress_line_interval_pct =
            Rcpp::as<double>(ctrl["progress_line_interval_pct"]);
    }
    if (ctrl.containsElementNamed("threads") && !Rf_isNull(ctrl["threads"])) {
        out.threads = Rcpp::as<int>(ctrl["threads"]);
    }
    if (ctrl.containsElementNamed("seed") && !Rf_isNull(ctrl["seed"])) {
        out.seed = Rcpp::as<long>(ctrl["seed"]);
    }
    if (ctrl.containsElementNamed("vector_storage")) {
        out.vector_storage = Rcpp::as<int>(ctrl["vector_storage"]);
    }
    if (ctrl.containsElementNamed("x_weights") && !Rf_isNull(ctrl["x_weights"])) {
        out.x_weights = Rcpp::as<std::vector<double>>(ctrl["x_weights"]);
    }
    if (ctrl.containsElementNamed("nlopt_params") &&
        !Rf_isNull(ctrl["nlopt_params"])) {
        Rcpp::List p = ctrl["nlopt_params"];
        if (p.hasAttribute("names")) {
            Rcpp::CharacterVector nms = p.names();
            for (int i = 0; i < p.size(); ++i) {
                out.nlopt_params[Rcpp::as<std::string>(nms[i])] =
                    Rcpp::as<double>(p[i]);
            }
        }
    }

    if (!include_em) return;

    if (ctrl.containsElementNamed("em_max_iter")) {
        out.em_max_iter = Rcpp::as<int>(ctrl["em_max_iter"]);
    }
    if (ctrl.containsElementNamed("em_tol")) {
        out.em_tol = Rcpp::as<double>(ctrl["em_tol"]);
    }
    if (ctrl.containsElementNamed("em_patience")) {
        out.em_patience = Rcpp::as<int>(ctrl["em_patience"]);
    }
    if (ctrl.containsElementNamed("em_init_mle")) {
        out.em_init_mle = Rcpp::as<bool>(ctrl["em_init_mle"]);
    }
    if (ctrl.containsElementNamed("tol")) {
        out.em_tol = Rcpp::as<double>(ctrl["tol"]);
    }
    if (ctrl.containsElementNamed("patience")) {
        out.em_patience = Rcpp::as<int>(ctrl["patience"]);
    }
    if (ctrl.containsElementNamed("diff")) {
        out.em_tol = Rcpp::as<double>(ctrl["diff"]);
    }
    if (ctrl.containsElementNamed("iter")) {
        Rcpp::NumericVector iter_vec(ctrl["iter"]);
        if (iter_vec.size() == 1) out.em_max_iter = static_cast<int>(iter_vec[0]);
        if (iter_vec.size() >= 2) out.em_max_iter = static_cast<int>(iter_vec[1]);
    }
}

inline Rcpp::List control_to_list(const NLoptControl& c, bool include_em) {
    Rcpp::RObject seed_obj = (c.seed >= 0) ? Rcpp::wrap(c.seed) : R_NilValue;
    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("algorithm") = c.algorithm,
        Rcpp::Named("local_algorithm") = c.local_algorithm,
        Rcpp::Named("maxeval") = c.maxeval,
        Rcpp::Named("xtol_rel") = c.xtol_rel,
        Rcpp::Named("xtol_abs") = c.xtol_abs,
        Rcpp::Named("ftol_rel") = c.ftol_rel,
        Rcpp::Named("ftol_abs") = c.ftol_abs,
        Rcpp::Named("maxtime") = c.maxtime,
        Rcpp::Named("stopval") = c.stopval,
        Rcpp::Named("population") = c.population,
        Rcpp::Named("initial_step") = c.initial_step,
        Rcpp::Named("print_level") = c.print_level,
        Rcpp::Named("threads") = c.threads,
        Rcpp::Named("x_weights") = c.x_weights,
        Rcpp::Named("seed") = seed_obj,
        Rcpp::Named("nlopt_params") = c.nlopt_params,
        Rcpp::Named("vector_storage") = c.vector_storage,
        Rcpp::Named("progress") = c.progress,
        Rcpp::Named("progress_refresh_ms") = c.progress_refresh_ms,
        Rcpp::Named("progress_line_interval_sec") =
            c.progress_line_interval_sec,
        Rcpp::Named("progress_line_interval_pct") =
            c.progress_line_interval_pct
    );
    if (include_em) {
        out["em_max_iter"] = c.em_max_iter;
        out["em_tol"] = c.em_tol;
        out["em_patience"] = c.em_patience;
        out["em_init_mle"] = c.em_init_mle;
    }
    return out;
}

} // namespace r_wrapper_modify_control

