#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../Cpp/include/estimate_mle.hpp"
#include "../../Cpp/include/modify_control.hpp"
#include "../../Cpp/include/progress_bar.hpp"
#define PY_MODIFY_OUT_IMPL "py_modify_outputs.cpp"
#include PY_MODIFY_OUT_IMPL
#define PY_CTRL_WRAP_IMPL "py_modify_control.cpp"
#include PY_CTRL_WRAP_IMPL

namespace {

// ============================
// Input Conversion Helpers
// ============================
// This helper converts Python dictionaries of parameter vectors into the
// C++ map format expected by the estimation core. The loop iterates over
// every key to preserve explicit field-level mapping.

void py_dict_to_cpp_map(
    const pybind11::dict& d,
    std::unordered_map<std::string, std::vector<double>>& out
) {
    for (auto item : d) {
        const std::string key = pybind11::str(item.first);
        out[key] = item.second.cast<std::vector<double>>();
    }
}


pybind11::list create_fit_rows(
    const std::vector<SubjectFitResult>& res_group,
    bool is_map,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    // The loop builds one output row per subject result.
    // We explicitly reconstruct flattened parameter keys in stable order
    // so downstream R/Python displays are deterministic.
    pybind11::list out_rows;
    for (const auto& r : res_group) {
        pybind11::dict row;
        row["subid"] = r.subid;
        row["logL"] = r.logL;
        if (is_map) {
            row["logPrior"] = r.logPrior;
            row["logPost"] = r.logPost;
        }
        row["aic"] = r.aic;
        row["bic"] = r.bic;
        row["status"] = r.status;

        auto flat_keys = py_modify_outputs::ordered_flat_names(ordered_params, param_sizes, r.best_params);
        for (const auto& key : flat_keys) {
            auto pos = key.rfind('_');
            std::string base = key;
            size_t idx = 0;
            bool is_indexed = false;
            if (pos != std::string::npos) {
                std::string tail = key.substr(pos + 1);
                bool all_digit = !tail.empty() && std::all_of(tail.begin(), tail.end(), ::isdigit);
                if (all_digit) {
                    base = key.substr(0, pos);
                    idx = static_cast<size_t>(std::stoul(tail) - 1);
                    is_indexed = true;
                }
            }
            auto it = r.best_params.find(base);
            if (it == r.best_params.end()) continue;
            std::vector<double> values = it->second;
            if (base == "c_conf" && !values.empty()) std::sort(values.begin(), values.end());
            if (!is_indexed && values.size() == 1) row[pybind11::str(key)] = values[0];
            else if (is_indexed && idx < values.size()) row[pybind11::str(key)] = values[idx];
        }
        out_rows.append(row);
    }
    return out_rows;
}

} // namespace

pybind11::dict py_estimate_mle(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const pybind11::dict& params,
    std::string model = "sdt",
    const pybind11::dict& control = pybind11::dict(),
    const pybind11::dict& lower = pybind11::dict(),
    const pybind11::dict& upper = pybind11::dict()
) {
    // ============================
    // Parameter Preparation
    // ============================
    // The wrapper first captures parameter order from user input. This
    // order is reused later to keep output columns stable and readable.
    ParamGroup user_params;
    std::vector<std::string> ordered_params;
    std::unordered_map<std::string, size_t> param_sizes;
    auto capture_order = [&](const pybind11::dict& d) {
        for (auto item : d) {
            const std::string key = pybind11::str(item.first);
            if (param_sizes.find(key) == param_sizes.end()) ordered_params.push_back(key);
            try { param_sizes[key] = item.second.cast<std::vector<double>>().size(); }
            catch (...) { param_sizes[key] = 1; }
        }
    };
    if (params.contains("free")) { pybind11::dict d = params["free"].cast<pybind11::dict>(); py_dict_to_cpp_map(d, user_params.free); capture_order(d); }
    if (params.contains("fixed")) { pybind11::dict d = params["fixed"].cast<pybind11::dict>(); py_dict_to_cpp_map(d, user_params.fixed); capture_order(d); }
    if (params.contains("constant")) { pybind11::dict d = params["constant"].cast<pybind11::dict>(); py_dict_to_cpp_map(d, user_params.constant); capture_order(d); }
    if (!params.contains("free") && !params.contains("fixed") && !params.contains("constant")) {
        py_dict_to_cpp_map(params, user_params.free);
        capture_order(params);
    }

    // ============================
    // Control Normalization
    // ============================
    // First apply user-provided control fields, then call modify_control()
    // so estimator defaults are injected in one place.
    NLoptControl cpp_control;
    py_wrapper_modify_control::apply_control_from_dict(
        control, cpp_control, false
    );
    cpp_control = modify_control(cpp_control, "mle");

    std::unordered_map<std::string, std::vector<double>> cpp_lower, cpp_upper;
    if (!lower.empty()) py_dict_to_cpp_map(lower, cpp_lower);
    if (!upper.empty()) py_dict_to_cpp_map(upper, cpp_upper);

    // ============================
    // Core Estimation Call
    // ============================
    // GIL is released during heavy C++ work to avoid blocking Python-side
    // scheduling and to allow OpenMP/threads to run without GIL contention.
    std::vector<SubjectFitResult> cpp_res;
    {
        pybind11::gil_scoped_release release;
        cpp_res = ::estimate_mle(df, colnames, user_params, model, cpp_control, cpp_lower, cpp_upper);
    }

    std::map<std::string, std::vector<SubjectFitResult>> grouped_res;
    for (const auto& r : cpp_res) grouped_res[r.cond].push_back(r);

    // ============================
    // Output Assembly
    // ============================
    // When condition is not split, fit is a single table-like list.
    // Otherwise, fit is split by condition and keyed by condition name.
    pybind11::object fit;
    if (grouped_res.size() == 1 && grouped_res.begin()->first == "") {
        fit = create_fit_rows(grouped_res.begin()->second, false, ordered_params, param_sizes);
    } else {
        pybind11::dict fit_by_cond;
        for (const auto& kv : grouped_res) fit_by_cond[pybind11::str(kv.first)] = create_fit_rows(kv.second, false, ordered_params, param_sizes);
        fit = std::move(fit_by_cond);
    }

    pybind11::dict estimator;
    estimator["name"] = "MLE";
    estimator["backend"] = "nlopt";
    estimator["global_algorithm"] = cpp_control.algorithm;
    estimator["local_algorithm"] = cpp_control.local_algorithm;
    estimator["control"] = py_wrapper_modify_control::control_to_dict(
        cpp_control, false
    );

    pybind11::list subject_opt;
    for (const auto& r : cpp_res) {
        pybind11::dict d;
        d["subid"] = r.subid;
        d["condition"] = r.cond;
        d["status"] = r.status;
        d["n_evals"] = r.n_evals;
        d["result_code"] = r.status;
        d["result_message"] = r.result_message;
        d["stop_reason"] = r.stop_reason;
        d["optimum_value"] = r.optimum_value;
        subject_opt.append(d);
    }

    ui::ProgressSnapshot ps = ui::progress_last_snapshot();
    pybind11::dict progress;
    progress["requested_mode"] = ps.requested_mode;
    progress["resolved_mode"] = ps.resolved_mode;
    progress["elapsed_sec"] = ps.elapsed_sec;
    progress["total_iterations"] = pybind11::int_(ps.total);
    progress["completed_iterations"] = pybind11::int_(ps.completed);
    progress["speed"] = ps.speed;
    progress["finished"] = ps.finished;

    pybind11::dict diagnostics;
    diagnostics["subject_optimization"] = subject_opt;
    diagnostics["progress"] = progress;

    pybind11::dict out;
    out["fit"] = fit;
    out["estimator"] = estimator;
    out["diagnostics"] = diagnostics;
    return out;
}

PYBIND11_MODULE(_estimate_mle, m) {
    m.doc() = "metaSDT: estimate_mle wrapper";
    m.def(
        "estimate_mle",
        &py_estimate_mle,
        pybind11::arg("df"),
        pybind11::arg("colnames"),
        pybind11::arg("params"),
        pybind11::arg("model") = "sdt",
        pybind11::arg("control") = pybind11::dict(),
        pybind11::arg("lower") = pybind11::dict(),
        pybind11::arg("upper") = pybind11::dict()
    );
}
