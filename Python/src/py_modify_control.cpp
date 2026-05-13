#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../Cpp/include/modify_control.hpp"

namespace py_wrapper_modify_control {

inline void apply_control_from_dict(
    const pybind11::dict& control,
    NLoptControl& out,
    bool include_em
) {
    if (control.contains("algorithm")) {
        out.algorithm = control["algorithm"].cast<std::string>();
    }
    if (control.contains("local_algorithm")) {
        out.local_algorithm = control["local_algorithm"].cast<std::string>();
    }
    if (control.contains("xtol_rel")) out.xtol_rel = control["xtol_rel"].cast<double>();
    if (control.contains("xtol_abs")) out.xtol_abs = control["xtol_abs"].cast<double>();
    if (control.contains("ftol_rel")) out.ftol_rel = control["ftol_rel"].cast<double>();
    if (control.contains("ftol_abs")) out.ftol_abs = control["ftol_abs"].cast<double>();
    if (control.contains("maxeval")) out.maxeval = control["maxeval"].cast<int>();
    if (control.contains("maxtime")) out.maxtime = control["maxtime"].cast<double>();
    if (control.contains("stopval")) out.stopval = control["stopval"].cast<double>();
    if (control.contains("population")) out.population = control["population"].cast<int>();
    if (control.contains("initial_step")) out.initial_step = control["initial_step"].cast<double>();
    if (control.contains("print_level") && !control["print_level"].is_none()) out.print_level = control["print_level"].cast<int>();
    if (control.contains("threads") && !control["threads"].is_none()) out.threads = control["threads"].cast<int>();
    if (control.contains("progress") && !control["progress"].is_none()) out.progress = control["progress"].cast<std::string>();
    if (control.contains("progress_refresh_ms")) out.progress_refresh_ms = control["progress_refresh_ms"].cast<int>();
    if (control.contains("progress_line_interval_sec")) out.progress_line_interval_sec = control["progress_line_interval_sec"].cast<double>();
    if (control.contains("progress_line_interval_pct")) out.progress_line_interval_pct = control["progress_line_interval_pct"].cast<double>();
    if (control.contains("seed") && !control["seed"].is_none()) out.seed = control["seed"].cast<long>();
    if (control.contains("vector_storage")) out.vector_storage = control["vector_storage"].cast<int>();
    if (control.contains("x_weights") && !control["x_weights"].is_none()) {
        out.x_weights = control["x_weights"].cast<std::vector<double>>();
    }
    if (control.contains("nlopt_params") && !control["nlopt_params"].is_none()) {
        pybind11::dict p = control["nlopt_params"].cast<pybind11::dict>();
        for (auto item : p) {
            out.nlopt_params[pybind11::str(item.first)] = item.second.cast<double>();
        }
    }

    if (!include_em) return;
    if (control.contains("em_max_iter")) out.em_max_iter = control["em_max_iter"].cast<int>();
    if (control.contains("em_tol")) out.em_tol = control["em_tol"].cast<double>();
    if (control.contains("em_patience")) out.em_patience = control["em_patience"].cast<int>();
    if (control.contains("em_init_mle")) out.em_init_mle = control["em_init_mle"].cast<bool>();
    if (control.contains("tol")) out.em_tol = control["tol"].cast<double>();
    if (control.contains("patience")) out.em_patience = control["patience"].cast<int>();
    if (control.contains("diff")) out.em_tol = control["diff"].cast<double>();
    if (control.contains("iter")) {
        std::vector<double> iter_vec = control["iter"].cast<std::vector<double>>();
        if (iter_vec.size() == 1) out.em_max_iter = static_cast<int>(iter_vec[0]);
        if (iter_vec.size() >= 2) out.em_max_iter = static_cast<int>(iter_vec[1]);
    }
}

inline pybind11::dict control_to_dict(const NLoptControl& c, bool include_em) {
    pybind11::dict out;
    out["algorithm"] = c.algorithm;
    out["local_algorithm"] = c.local_algorithm;
    out["maxeval"] = c.maxeval;
    out["xtol_rel"] = c.xtol_rel;
    out["xtol_abs"] = c.xtol_abs;
    out["ftol_rel"] = c.ftol_rel;
    out["ftol_abs"] = c.ftol_abs;
    out["maxtime"] = c.maxtime;
    out["stopval"] = c.stopval;
    out["population"] = c.population;
    out["initial_step"] = c.initial_step;
    out["print_level"] = c.print_level;
    out["threads"] = c.threads;
    out["x_weights"] = c.x_weights;
    out["seed"] = (c.seed >= 0) ? pybind11::cast(c.seed) : pybind11::none();
    out["nlopt_params"] = c.nlopt_params;
    out["vector_storage"] = c.vector_storage;
    out["progress"] = c.progress;
    out["progress_refresh_ms"] = c.progress_refresh_ms;
    out["progress_line_interval_sec"] = c.progress_line_interval_sec;
    out["progress_line_interval_pct"] = c.progress_line_interval_pct;
    if (include_em) {
        out["em_max_iter"] = c.em_max_iter;
        out["em_tol"] = c.em_tol;
        out["em_patience"] = c.em_patience;
        out["em_init_mle"] = c.em_init_mle;
    }
    return out;
}

} // namespace py_wrapper_modify_control

