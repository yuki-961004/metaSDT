#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../Cpp/include/estimate_map.hpp"
#include "../../Cpp/include/modify_control.hpp"
#include "../../Cpp/include/modify_prior.hpp"
#include "../../Cpp/include/progress_bar.hpp"
#define PY_MODIFY_OUT_IMPL "py_modify_outputs.cpp"
#include PY_MODIFY_OUT_IMPL
#define PY_CTRL_WRAP_IMPL "py_modify_control.cpp"
#include PY_CTRL_WRAP_IMPL

namespace {

// ============================
// Input Conversion Helpers
// ============================
// These helper functions convert Python input structures into C++ core
// structures. We keep conversion explicit to guarantee field mapping and
// predictable behavior across Python versions.

void py_dict_to_cpp_map(
    const pybind11::dict& d,
    std::unordered_map<std::string, std::vector<double>>& out
) {
    for (auto item : d) {
        const std::string key = pybind11::str(item.first);
        out[key] = item.second.cast<std::vector<double>>();
    }
}

void py_dict_to_user_priors(
    const pybind11::dict& d,
    std::unordered_map<std::string, UserPrior>& out
) {
    for (auto item : d) {
        const std::string param_name = pybind11::str(item.first);
        pybind11::dict prior_dict = item.second.cast<pybind11::dict>();
        UserPrior up;
        up.type = prior_dict.contains("type") ? prior_dict["type"].cast<std::string>() : "none";
        for (auto arg_item : prior_dict) {
            const std::string arg_key = pybind11::str(arg_item.first);
            if (arg_key != "type") up.args[arg_key] = arg_item.second.cast<double>();
        }
        out[param_name] = up;
    }
}

pybind11::list create_fit_rows(
    const std::vector<SubjectFitResult>& res_group,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    // Build one row per subject and flatten parameter vectors into stable
    // key names (`name`, `name_1`, `name_2`, ...).
    pybind11::list out_rows;
    for (const auto& r : res_group) {
        pybind11::dict row;
        row["subid"] = r.subid;
        row["logL"] = r.logL;
        row["logPrior"] = r.logPrior;
        row["logPost"] = r.logPost;
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

pybind11::dict summarize_distribution(const std::string& type, const std::vector<double>& vals) {
    pybind11::dict out;
    if (vals.empty()) {
        out["distribution"] = type;
        out["parameters"] = pybind11::dict();
        return out;
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
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (t == "norm") t = "normal";
    if (t == "lnorm") t = "lognormal";
    if (t == "exp") t = "exponential";
    if (t == "unif") t = "uniform";

    pybind11::dict p;
    if (t == "normal") {
        p["mean"] = mean;
        p["sd"] = std::sqrt(var);
    } else if (t == "exponential") {
        p["rate"] = 1.0 / std::max(mean, 1e-6);
    } else if (t == "uniform") {
        auto mm = std::minmax_element(vals.begin(), vals.end());
        p["min"] = *mm.first;
        p["max"] = *mm.second;
    } else if (t == "lognormal") {
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
        p["meanlog"] = lmean;
        p["sdlog"] = std::sqrt(lvar);
    } else if (t == "beta") {
        double m = std::max(1e-4, std::min(mean, 1.0 - 1e-4));
        double max_var = m * (1.0 - m) - 1e-6;
        double v = std::min(var, max_var);
        double common = (m * (1.0 - m) / std::max(v, 1e-8)) - 1.0;
        p["shape1"] = std::max(m * common, 1e-3);
        p["shape2"] = std::max((1.0 - m) * common, 1e-3);
    } else if (t == "cauchy") {
        std::vector<double> x = vals;
        std::sort(x.begin(), x.end());
        const size_t n = x.size();
        const double location = (n % 2 == 0) ? (x[n / 2 - 1] + x[n / 2]) / 2.0 : x[n / 2];
        const double scale = std::max((x[(3 * n) / 4] - x[n / 4]) / 2.0, 1e-4);
        p["location"] = location;
        p["scale"] = scale;
    } else {
        t = "none";
    }

    out["distribution"] = t;
    out["parameters"] = p;
    return out;
}

pybind11::dict summarize_posteriors(
    const std::vector<SubjectFitResult>& cpp_res,
    const std::unordered_map<std::string, UserPrior>& user_priors,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    pybind11::dict out;
    if (cpp_res.empty()) return out;

    std::unordered_map<std::string, UserPrior> prior_map = default_priors();
    for (const auto& kv : user_priors) prior_map[kv.first] = kv.second;

    std::vector<std::string> base_order = py_modify_outputs::ordered_base_names(ordered_params, cpp_res[0].best_params);

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
            const std::string key = (p_len == 1) ? base : (base + "_" + std::to_string(j + 1));
            out[pybind11::str(key)] = summarize_distribution(type, vals);
        }
    }
    return out;
}

pybind11::dict build_solution(
    const std::vector<SubjectFitResult>& cond_res,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    pybind11::dict out;
    if (cond_res.empty()) return out;

    std::vector<std::string> base_order = py_modify_outputs::ordered_base_names(ordered_params, cond_res[0].best_params);

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
            const std::string key = (p_len == 1) ? base : (base + "_" + std::to_string(j + 1));
            if (vals.empty()) out[pybind11::str(key)] = pybind11::none();
            else out[pybind11::str(key)] = vals;
        }
    }
    return out;
}

pybind11::dict build_condition_diagnostics(
    const std::vector<SubjectFitResult>& cond_res,
    const std::unordered_map<std::string, UserPrior>& cpp_priors,
    int em_iterations,
    const std::string& em_stop_reason,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    pybind11::list subject_opt;
    for (const auto& r : cond_res) {
        pybind11::dict d;
        d["subid"] = r.subid;
        d["status"] = r.status;
        d["n_evals"] = r.n_evals;
        d["result_code"] = r.status;
        d["result_message"] = r.result_message;
        d["stop_reason"] = r.stop_reason;
        d["optimum_value"] = r.optimum_value;
        subject_opt.append(d);
    }

    pybind11::dict out;
    out["em_iterations"] = em_iterations;
    out["em_stop_reason"] = em_stop_reason;
    out["subject_optimization"] = subject_opt;
    out["posterior"] = summarize_posteriors(cond_res, cpp_priors, ordered_params, param_sizes);
    out["solution"] = build_solution(cond_res, ordered_params, param_sizes);
    return out;
}

} // namespace

pybind11::dict py_estimate_map(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const pybind11::dict& params,
    std::string model = "sdt",
    const pybind11::dict& control = pybind11::dict(),
    const pybind11::dict& lower = pybind11::dict(),
    const pybind11::dict& upper = pybind11::dict(),
    const pybind11::dict& user_priors = pybind11::dict()
) {
    // ============================
    // Parameter Preparation
    // ============================
    // We record both user-provided parameter order and vector lengths.
    // This order later drives posterior/solution key ordering.
    ParamGroup user_params;
    std::vector<std::string> ordered_params;
    std::unordered_map<std::string, size_t> param_sizes;
    auto capture_order = [&](const pybind11::dict& d) {
        for (auto item : d) {
            const std::string key = pybind11::str(item.first);
            if (param_sizes.find(key) == param_sizes.end()) ordered_params.push_back(key);
            try {
                param_sizes[key] = item.second.cast<std::vector<double>>().size();
            } catch (...) {
                param_sizes[key] = 1;
            }
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
    // First copy explicit user values, then merge estimator defaults with
    // modify_control("map") to ensure consistency with R-side behavior.
    NLoptControl cpp_control;
    py_wrapper_modify_control::apply_control_from_dict(control, cpp_control, true);
    cpp_control = modify_control(cpp_control, "map");

    std::unordered_map<std::string, std::vector<double>> cpp_lower, cpp_upper;
    if (!lower.empty()) py_dict_to_cpp_map(lower, cpp_lower);
    if (!upper.empty()) py_dict_to_cpp_map(upper, cpp_upper);

    std::unordered_map<std::string, UserPrior> cpp_priors;
    if (!user_priors.empty()) py_dict_to_user_priors(user_priors, cpp_priors);

    // ============================
    // Core MAP Estimation
    // ============================
    // MAP may execute multiple EM-like outer iterations. We capture both
    // global and per-condition iteration diagnostics from the core API.
    int em_iterations_used = 0;
    std::unordered_map<std::string, int> em_by_cond;
    std::unordered_map<std::string, std::string> stop_by_cond;

    std::vector<SubjectFitResult> cpp_res;
    {
        pybind11::gil_scoped_release release;
        cpp_res = ::estimate_map(
            df, colnames, user_params, model, cpp_control, cpp_lower, cpp_upper, cpp_priors,
            &em_iterations_used, &em_by_cond, &stop_by_cond
        );
    }

    std::map<std::string, std::vector<SubjectFitResult>> grouped_res;
    for (const auto& r : cpp_res) grouped_res[r.cond].push_back(r);

    // ============================
    // Output Assembly
    // ============================
    // Single condition returns one table-like list; multi-condition returns
    // a keyed dictionary to preserve condition split semantics.
    pybind11::object fit;
    if (grouped_res.size() == 1 && grouped_res.begin()->first == "") {
        fit = create_fit_rows(grouped_res.begin()->second, ordered_params, param_sizes);
    } else {
        pybind11::dict fit_by_cond;
        for (const auto& kv : grouped_res) fit_by_cond[pybind11::str(kv.first)] = create_fit_rows(kv.second, ordered_params, param_sizes);
        fit = std::move(fit_by_cond);
    }

    pybind11::dict estimator;
    estimator["name"] = "MAP";
    estimator["backend"] = "nlopt";
    estimator["global_algorithm"] = cpp_control.algorithm;
    estimator["local_algorithm"] = cpp_control.local_algorithm;
    estimator["control"] = py_wrapper_modify_control::control_to_dict(cpp_control, true);

    pybind11::dict diagnostics;
    if (grouped_res.size() == 1 && grouped_res.begin()->first == "") {
        int em_iter = em_iterations_used;
        std::string reason = "unknown";
        auto it_i = em_by_cond.find("");
        if (it_i != em_by_cond.end()) em_iter = it_i->second;
        auto it_r = stop_by_cond.find("");
        if (it_r != stop_by_cond.end()) reason = it_r->second;
        diagnostics = build_condition_diagnostics(grouped_res.begin()->second, cpp_priors, em_iter, reason, ordered_params, param_sizes);
    } else {
        for (const auto& kv : grouped_res) {
            int em_iter = 0;
            std::string reason = "unknown";
            auto it_i = em_by_cond.find(kv.first);
            if (it_i != em_by_cond.end()) em_iter = it_i->second;
            auto it_r = stop_by_cond.find(kv.first);
            if (it_r != stop_by_cond.end()) reason = it_r->second;
            diagnostics[pybind11::str(kv.first)] = build_condition_diagnostics(kv.second, cpp_priors, em_iter, reason, ordered_params, param_sizes);
        }
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

    if (grouped_res.size() == 1 && grouped_res.begin()->first == "") {
        diagnostics["progress"] = progress;
    } else {
        diagnostics["__progress__"] = progress;
    }

    pybind11::dict out;
    out["fit"] = fit;
    out["estimator"] = estimator;
    out["diagnostics"] = diagnostics;
    return out;
}

PYBIND11_MODULE(_estimate_map, m) {
    m.doc() = "metaSDT: estimate_map wrapper";
    m.def(
        "estimate_map",
        &py_estimate_map,
        pybind11::arg("df"),
        pybind11::arg("colnames"),
        pybind11::arg("params"),
        pybind11::arg("model") = "sdt",
        pybind11::arg("control") = pybind11::dict(),
        pybind11::arg("lower") = pybind11::dict(),
        pybind11::arg("upper") = pybind11::dict(),
        pybind11::arg("user_priors") = pybind11::dict()
    );
}
