#include <Rcpp.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../../Cpp/include/shell_run_m.hpp"

#define CORE_IMPL_SHELL_RUN_M "../../Cpp/src/shell_run_m.cpp"
#include CORE_IMPL_SHELL_RUN_M

namespace {

/* ========================================================================== *
 *                              R Conversion                                  *
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

    Rcpp::List list(robj);
    for (int i = 0; i < list.size(); ++i) {
        cpp_map[Rcpp::as<std::string>(names[i])] =
            Rcpp::as<std::vector<double>>(list[i]);
    }
}

ParamGroup params_to_cpp(Rcpp::RObject params) {
    ParamGroup out;
    if (params.isNULL() || Rf_length(params) == 0) {
        return out;
    }
    if (!Rcpp::is<Rcpp::List>(params)) {
        Rcpp::stop("params must be a list or NULL.");
    }

    Rcpp::List list(params);
    const bool structured = list.containsElementNamed("free") ||
        list.containsElementNamed("fixed") ||
        list.containsElementNamed("constant");

    if (structured) {
        if (list.containsElementNamed("free")) {
            r_obj_to_cpp_map(list["free"], out.free);
        }
        if (list.containsElementNamed("fixed")) {
            r_obj_to_cpp_map(list["fixed"], out.fixed);
        }
        if (list.containsElementNamed("constant")) {
            r_obj_to_cpp_map(list["constant"], out.constant);
        }
    } else {
        r_obj_to_cpp_map(params, out.free);
    }

    return out;
}

ShellRunMOptions option_to_cpp(Rcpp::Nullable<Rcpp::List> option) {
    ShellRunMOptions out;
    if (option.isNull()) {
        return out;
    }

    Rcpp::List opt(option);
    if (opt.containsElementNamed("n") && !Rf_isNull(opt["n"])) {
        out.n = Rcpp::as<int>(opt["n"]);
    }
    if (opt.containsElementNamed("seed") && !Rf_isNull(opt["seed"])) {
        out.seed = Rcpp::as<int>(opt["seed"]);
        out.has_seed = true;
    }
    if (opt.containsElementNamed("density_points") &&
        !Rf_isNull(opt["density_points"])) {
        out.density_points = Rcpp::as<int>(opt["density_points"]);
    }
    if (opt.containsElementNamed("xlim") && !Rf_isNull(opt["xlim"])) {
        out.xlim = Rcpp::as<std::vector<double>>(opt["xlim"]);
        out.has_xlim = true;
    }

    return out;
}

Rcpp::DataFrame data_to_frame(const ShellRunMData& data) {
    return Rcpp::DataFrame::create(
        Rcpp::Named("trial") = data.trial,
        Rcpp::Named("stim") = data.stim,
        Rcpp::Named("resp") = data.resp,
        Rcpp::Named("conf") = data.conf,
        Rcpp::Named("diff") = data.diff,
        Rcpp::Named("evidence") = data.evidence
    );
}

Rcpp::DataFrame density_to_frame(const ShellRunMDensity& density) {
    return Rcpp::DataFrame::create(
        Rcpp::Named("x") = density.x,
        Rcpp::Named("noise") = density.noise,
        Rcpp::Named("signal") = density.signal
    );
}

} // namespace

/* ========================================================================== *
 *                              R Entry Point                                 *
 * ========================================================================== */

// [[Rcpp::export(name = ".shell_run_m_cpp")]]
Rcpp::List r_shell_run_m(
    Rcpp::RObject params,
    std::string model = "sdt",
    Rcpp::Nullable<Rcpp::List> option = R_NilValue
) {
    const ParamGroup cpp_params = params_to_cpp(params);
    const ShellRunMOptions cpp_option = option_to_cpp(option);
    const ShellRunMResult result = shell_run_m(cpp_params, model, cpp_option);

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("params") = Rcpp::wrap(result.params),
        Rcpp::Named("model") = result.model,
        Rcpp::Named("data") = data_to_frame(result.data),
        Rcpp::Named("density") = density_to_frame(result.density),
        Rcpp::Named("criteria") = result.criteria,
        Rcpp::Named("seed") = result.seed
    );

    return out;
}
