#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../Cpp/include/estimate_mcmc.hpp"
#include "../../Cpp/include/modify_control.hpp"
#include "../../Cpp/include/modify_outputs.hpp"
#include "../../Cpp/include/modify_prior.hpp"
#include "../../Cpp/include/progress_bar.hpp"
#define PY_MODIFY_OUT_IMPL "py_modify_outputs.cpp"
#include PY_MODIFY_OUT_IMPL
#define PY_CTRL_WRAP_IMPL "py_modify_control.cpp"
#include PY_CTRL_WRAP_IMPL

namespace {

/* ========================================================================== *
 *                            Input Conversion                                *
 * ========================================================================== */

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

        UserPrior prior;
        prior.type = prior_dict.contains("type")
            ? prior_dict["type"].cast<std::string>()
            : "none";

        // type 以外的键都是分布参数, 逐一写入 C++ prior 结构
        for (auto arg_item : prior_dict) {
            const std::string arg_key = pybind11::str(arg_item.first);
            if (arg_key != "type") {
                prior.args[arg_key] = arg_item.second.cast<double>();
            }
        }

        out[param_name] = prior;
    }
}

/* ========================================================================== *
 *                             Output Helpers                                 *
 * ========================================================================== */

bool flat_key_value(
    const std::unordered_map<std::string, std::vector<double>>& params,
    const std::string& key,
    double& value
) {
    auto direct = params.find(key);
    if (direct != params.end() && direct->second.size() == 1) {
        value = direct->second[0];
        return true;
    }

    const size_t pos = key.rfind('_');
    if (pos == std::string::npos) {
        return false;
    }

    const std::string tail = key.substr(pos + 1);
    const bool all_digit = !tail.empty() &&
        std::all_of(
            tail.begin(),
            tail.end(),
            [](unsigned char ch) {
                return std::isdigit(ch) != 0;
            }
        );
    if (!all_digit) {
        return false;
    }

    const std::string base = key.substr(0, pos);
    const size_t index = static_cast<size_t>(std::stoul(tail) - 1);
    auto found = params.find(base);
    if (found == params.end() || index >= found->second.size()) {
        return false;
    }

    std::vector<double> values = found->second;
    if (base == "c_conf" && !values.empty()) {
        std::sort(values.begin(), values.end());
    }
    value = values[index];
    return true;
}

pybind11::dict samples_to_dict(const OutputDiagnosticsRow& row) {
    pybind11::dict out;

    for (const auto& kv : row.samples) {
        out[pybind11::str(kv.first)] = kv.second;
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

pybind11::list create_fit_rows(
    const std::vector<OutputFitRow>& rows,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    pybind11::list out_rows;

    std::vector<std::string> param_names;
    std::vector<std::string> sd_names;
    if (!rows.empty()) {
        param_names = py_modify_outputs::ordered_flat_names(
            ordered_params,
            param_sizes,
            rows[0].params
        );
        sd_names = py_modify_outputs::ordered_flat_names(
            ordered_params,
            param_sizes,
            rows[0].param_sd
        );
    }

    for (const OutputFitRow& row : rows) {
        pybind11::dict item;
        item["subid"] = row.subid;
        item["logL"] = row.logL;
        item["logPrior"] = row.logPrior;
        item["logPost"] = row.logPost;
        item["aic"] = row.aic;
        item["bic"] = row.bic;
        item["status"] = row.status;

        for (const std::string& key : param_names) {
            double value = 0.0;
            if (flat_key_value(row.params, key, value)) {
                item[pybind11::str(key)] = value;
            }
        }

        for (const std::string& key : sd_names) {
            double value = 0.0;
            if (flat_key_value(row.param_sd, key, value)) {
                item[pybind11::str(key + "_sd")] = value;
            }
        }

        out_rows.append(item);
    }

    return out_rows;
}

pybind11::object create_fit_slot(
    const OutputFitSlot& slot,
    const std::vector<std::string>& ordered_params,
    const std::unordered_map<std::string, size_t>& param_sizes
) {
    if (slot.by_condition.size() == 1 &&
        slot.by_condition.begin()->first.empty()) {
        return create_fit_rows(
            slot.by_condition.begin()->second,
            ordered_params,
            param_sizes
        );
    }

    pybind11::dict out;
    for (const auto& kv : slot.by_condition) {
        out[pybind11::str(kv.first)] = create_fit_rows(
            kv.second,
            ordered_params,
            param_sizes
        );
    }
    return std::move(out);
}

pybind11::list create_subject_sampling(
    const std::vector<OutputDiagnosticsRow>& rows
) {
    pybind11::list out;

    for (const OutputDiagnosticsRow& row : rows) {
        pybind11::dict item;
        item["subid"] = row.subid;
        item["condition"] = row.cond;
        item["status"] = row.status;
        item["n_evals"] = row.n_evals;
        item["result_code"] = row.result_code;
        item["n_draws"] = row.n_draws;
        item["n_chains"] = row.n_chains;
        item["warmup"] = row.warmup;
        item["thin"] = row.thin;
        item["leapfrog_steps"] = row.leapfrog_steps;
        item["accept_rate"] = row.accept_rate;
        item["step_size"] = row.step_size;
        item["result_message"] = row.result_message;
        item["stop_reason"] = row.stop_reason;
        out.append(item);
    }

    return out;
}

pybind11::dict create_condition_diagnostics(
    const std::vector<OutputDiagnosticsRow>& rows
) {
    pybind11::dict posterior_samples;

    // posterior_samples 可能较大, 因此放 diagnostics 的独立子槽
    for (const OutputDiagnosticsRow& row : rows) {
        posterior_samples[pybind11::str(subject_key(row))] =
            samples_to_dict(row);
    }

    pybind11::dict out;
    out["subject_sampling"] = create_subject_sampling(rows);
    out["posterior_samples"] = posterior_samples;
    return out;
}

pybind11::dict progress_to_dict() {
    ui::ProgressSnapshot ps = ui::progress_last_snapshot();
    pybind11::dict out;
    out["requested_mode"] = ps.requested_mode;
    out["resolved_mode"] = ps.resolved_mode;
    out["elapsed_sec"] = ps.elapsed_sec;
    out["total_iterations"] = pybind11::int_(ps.total);
    out["completed_iterations"] = pybind11::int_(ps.completed);
    out["speed"] = ps.speed;
    out["finished"] = ps.finished;
    return out;
}

pybind11::dict create_diagnostics_slot(const OutputDiagnosticsSlot& slot) {
    pybind11::dict out;

    if (slot.by_condition.size() == 1 &&
        slot.by_condition.begin()->first.empty()) {
        out = create_condition_diagnostics(
            slot.by_condition.begin()->second
        );
        out["progress"] = progress_to_dict();
        return out;
    }

    for (const auto& kv : slot.by_condition) {
        out[pybind11::str(kv.first)] = create_condition_diagnostics(
            kv.second
        );
    }
    out["__progress__"] = progress_to_dict();
    return out;
}

} // namespace

pybind11::dict py_estimate_mcmc(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const pybind11::dict& params,
    std::string model = "sdt",
    const pybind11::dict& control = pybind11::dict(),
    const pybind11::dict& lower = pybind11::dict(),
    const pybind11::dict& upper = pybind11::dict(),
    const pybind11::dict& priors = pybind11::dict()
) {
    ParamGroup user_params;
    std::vector<std::string> ordered_params;
    std::unordered_map<std::string, size_t> param_sizes;

    auto capture_order = [&](const pybind11::dict& d) {
        for (auto item : d) {
            const std::string key = pybind11::str(item.first);
            if (param_sizes.find(key) == param_sizes.end()) {
                ordered_params.push_back(key);
            }
            try {
                param_sizes[key] =
                    item.second.cast<std::vector<double>>().size();
            } catch (...) {
                param_sizes[key] = 1;
            }
        }
    };

    if (params.contains("free")) {
        pybind11::dict d = params["free"].cast<pybind11::dict>();
        py_dict_to_cpp_map(d, user_params.free);
        capture_order(d);
    }
    if (params.contains("fixed")) {
        pybind11::dict d = params["fixed"].cast<pybind11::dict>();
        py_dict_to_cpp_map(d, user_params.fixed);
        capture_order(d);
    }
    if (params.contains("constant")) {
        pybind11::dict d = params["constant"].cast<pybind11::dict>();
        py_dict_to_cpp_map(d, user_params.constant);
        capture_order(d);
    }
    if (!params.contains("free") &&
        !params.contains("fixed") &&
        !params.contains("constant")) {
        py_dict_to_cpp_map(params, user_params.free);
        capture_order(params);
    }

    StanControl cpp_control;
    py_wrapper_modify_control::apply_control_from_dict(
        control,
        cpp_control
    );
    cpp_control = modify_control(cpp_control, "mcmc");

    std::unordered_map<std::string, std::vector<double>> cpp_lower;
    std::unordered_map<std::string, std::vector<double>> cpp_upper;
    if (!lower.empty()) {
        py_dict_to_cpp_map(lower, cpp_lower);
    }
    if (!upper.empty()) {
        py_dict_to_cpp_map(upper, cpp_upper);
    }

    std::unordered_map<std::string, UserPrior> cpp_priors;
    if (!priors.empty()) {
        py_dict_to_user_priors(priors, cpp_priors);
    }

    std::vector<SubjectMCMCResult> cpp_res;
    {
        pybind11::gil_scoped_release release;
        cpp_res = ::estimate_mcmc(
            df,
            colnames,
            user_params,
            model,
            cpp_control,
            cpp_lower,
            cpp_upper,
            cpp_priors
        );
    }

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

    pybind11::dict estimator;
    estimator["name"] = estimator_slot.name;
    estimator["backend"] = estimator_slot.backend;
    estimator["algorithm"] = estimator_slot.algorithm;
    estimator["method"] = estimator_slot.algorithm;
    estimator["global_algorithm"] = estimator_slot.global_algorithm;
    estimator["local_algorithm"] = estimator_slot.local_algorithm;
    estimator["control"] =
        py_wrapper_modify_control::control_to_dict(cpp_control);

    pybind11::dict out;
    out["fit"] = create_fit_slot(
        output.fit,
        ordered_params,
        param_sizes
    );
    out["estimator"] = estimator;
    out["diagnostics"] = create_diagnostics_slot(output.diagnostics);
    return out;
}

PYBIND11_MODULE(_estimate_mcmc, m) {
    m.doc() = "metaSDT: estimate_mcmc wrapper";
    m.def(
        "estimate_mcmc",
        &py_estimate_mcmc,
        pybind11::arg("df"),
        pybind11::arg("colnames"),
        pybind11::arg("params"),
        pybind11::arg("model") = "sdt",
        pybind11::arg("control") = pybind11::dict(),
        pybind11::arg("lower") = pybind11::dict(),
        pybind11::arg("upper") = pybind11::dict(),
        pybind11::arg("priors") = pybind11::dict()
    );
}
