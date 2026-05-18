#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../Cpp/include/estimate_abc.hpp"

namespace py = pybind11;

namespace {

bool is_sequence_like(const py::handle& obj) {
    return py::isinstance<py::sequence>(obj) &&
           !py::isinstance<py::str>(obj) &&
           !py::isinstance<py::bytes>(obj);
}

bool sequence_is_nested(const py::sequence& seq) {
    if (seq.empty()) {
        return false;
    }
    return is_sequence_like(seq[0]);
}

std::size_t infer_n_samples(const py::dict& params) {
    std::size_t n = 0;
    for (auto item : params) {
        py::handle value = item.second;
        if (!is_sequence_like(value)) {
            continue;
        }
        py::sequence seq = value.cast<py::sequence>();
        if (seq.empty()) {
            continue;
        }
        const std::size_t candidate = seq.size();
        if (n == 0) {
            n = candidate;
        } else if (candidate != 1 && candidate != n) {
            throw std::invalid_argument(
                "ABC parameter columns must have equal length or length 1."
            );
        }
    }
    return n == 0 ? 1 : n;
}

std::vector<double> values_for_sample(
    const py::handle& value,
    std::size_t index,
    std::size_t n_samples
) {
    if (!is_sequence_like(value)) {
        return {value.cast<double>()};
    }

    py::sequence seq = value.cast<py::sequence>();
    if (seq.empty()) {
        return {};
    }

    if (sequence_is_nested(seq)) {
        const std::size_t row_index = seq.size() == 1 ? 0 : index;
        if (seq.size() != 1 && seq.size() != n_samples) {
            throw std::invalid_argument(
                "Nested ABC parameter columns must have n_samples rows."
            );
        }
        return seq[row_index].cast<std::vector<double>>();
    }

    if (seq.size() == n_samples) {
        return {seq[index].cast<double>()};
    }

    if (seq.size() == 1) {
        return {seq[0].cast<double>()};
    }

    return seq.cast<std::vector<double>>();
}

void py_dict_to_cpp_map(
    const py::dict& d,
    std::unordered_map<std::string, std::vector<double>>& out
) {
    for (auto item : d) {
        const std::string key = py::str(item.first);
        try {
            out[key] = item.second.cast<std::vector<double>>();
        } catch (...) {
            out[key] = {item.second.cast<double>()};
        }
    }
}

void py_dict_to_user_priors(
    const py::dict& d,
    std::unordered_map<std::string, UserPrior>& out
) {
    for (auto item : d) {
        const std::string key = py::str(item.first);
        py::dict spec = item.second.cast<py::dict>();
        UserPrior prior;
        prior.type = spec.contains("type")
            ? py::str(spec["type"]).cast<std::string>()
            : "none";
        for (auto arg : spec) {
            const std::string arg_key = py::str(arg.first);
            if (arg_key != "type") {
                prior.args[arg_key] = arg.second.cast<double>();
            }
        }
        out[key] = prior;
    }
}

std::vector<std::unordered_map<std::string, std::vector<double>>> parse_param_samples(
    const py::dict& params
) {
    const std::size_t n_samples = infer_n_samples(params);
    std::vector<std::unordered_map<std::string, std::vector<double>>> out(n_samples);
    for (auto item : params) {
        const std::string key = py::str(item.first);
        for (std::size_t i = 0; i < n_samples; ++i) {
            out[i][key] = values_for_sample(item.second, i, n_samples);
        }
    }
    return out;
}

py::dict control_to_dict(const ABCControl& control) {
    py::dict out;
    out["tol"] = control.tol;
    out["method"] = control.method;
    out["reduction"] = control.reduction;
    out["n_comp"] = control.n_comp;
    out["samples"] = control.samples;
    out["kernel"] = control.kernel;
    out["hcorr"] = control.hcorr;
    out["seed"] = control.seed;
    out["print_level"] = control.print_level;
    return out;
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

ABCControl parse_control(const py::dict& control) {
    ABCControl out;
    if (control.contains("tol")) out.tol = control["tol"].cast<double>();
    if (control.contains("method")) out.method = control["method"].cast<std::string>();
    if (control.contains("reduction")) out.reduction = control["reduction"].cast<std::string>();
    if (control.contains("n_comp")) out.n_comp = control["n_comp"].cast<int>();
    if (control.contains("ncomp")) out.n_comp = control["ncomp"].cast<int>();
    if (control.contains("samples")) out.samples = control["samples"].cast<int>();
    if (control.contains("kernel")) out.kernel = control["kernel"].cast<std::string>();
    if (control.contains("hcorr")) out.hcorr = control["hcorr"].cast<bool>();
    if (control.contains("seed")) out.seed = control["seed"].cast<unsigned int>();
    if (control.contains("print_level")) out.print_level = control["print_level"].cast<int>();
    return out;
}

py::list create_fit_rows(const std::vector<SubjectABCResult>& results) {
    py::list rows;
    for (const auto& result : results) {
        py::dict row;
        row["subid"] = result.subid;
        row["status"] = result.status;
        const std::size_t n = std::min(
            result.parameter_names.size(),
            result.summary.size()
        );
        for (std::size_t i = 0; i < n; ++i) {
            row[py::str(result.parameter_names[i])] = result.summary[i].mean;
        }
        rows.append(row);
    }
    return rows;
}

py::dict create_summary_rows(const std::vector<SubjectABCResult>& results) {
    py::dict out;
    for (const auto& result : results) {
        py::list rows;
        const std::size_t n = std::min(
            result.parameter_names.size(),
            result.summary.size()
        );
        for (std::size_t i = 0; i < n; ++i) {
            const auto& s = result.summary[i];
            py::dict row;
            row["parameter"] = result.parameter_names[i];
            row["min"] = s.min;
            row["q_lower"] = s.q_lower;
            row["median"] = s.median;
            row["mean"] = s.mean;
            row["mode"] = s.mode;
            row["q_upper"] = s.q_upper;
            row["max"] = s.max;
            row["sd"] = s.sd;
            rows.append(row);
        }
        const std::string key = result.cond.empty()
            ? std::to_string(result.subid)
            : std::to_string(result.subid) + ":" + result.cond;
        out[py::str(key)] = rows;
    }
    return out;
}

} // namespace

py::dict py_estimate_abc(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const py::dict& params,
    std::string model = "sdt",
    const py::dict& control = py::dict(),
    const py::dict& priors = py::dict()
) {
    ParamGroup user_params;
    if (params.contains("free")) {
        py_dict_to_cpp_map(params["free"].cast<py::dict>(), user_params.free);
    }
    if (params.contains("fixed")) {
        py_dict_to_cpp_map(params["fixed"].cast<py::dict>(), user_params.fixed);
    }
    if (params.contains("constant")) {
        py_dict_to_cpp_map(params["constant"].cast<py::dict>(), user_params.constant);
    }
    if (!params.contains("free") &&
        !params.contains("fixed") &&
        !params.contains("constant")) {
        py_dict_to_cpp_map(params, user_params.free);
    }

    std::unordered_map<std::string, UserPrior> cpp_priors;
    if (!priors.empty()) {
        py_dict_to_user_priors(priors, cpp_priors);
    }

    const ABCControl cpp_control = parse_control(control);

    std::vector<SubjectABCResult> cpp_res;
    {
        py::gil_scoped_release release;
        cpp_res = ::estimate_abc(
            df,
            colnames,
            user_params,
            model,
            cpp_control,
            cpp_priors
        );
    }

    std::map<std::string, std::vector<SubjectABCResult>> grouped;
    for (const auto& r : cpp_res) {
        grouped[r.cond].push_back(r);
    }

    py::object fit;
    if (grouped.size() == 1 && grouped.begin()->first.empty()) {
        fit = create_fit_rows(grouped.begin()->second);
    } else {
        py::dict by_cond;
        for (const auto& kv : grouped) {
            by_cond[py::str(kv.first)] = create_fit_rows(kv.second);
        }
        fit = std::move(by_cond);
    }

    py::list subject_abc;
    for (const auto& r : cpp_res) {
        py::dict row;
        row["subid"] = r.subid;
        row["condition"] = r.cond;
        row["status"] = r.status;
        row["message"] = r.message;
        row["n_comp"] = r.n_comp_used;
        row["n_accepted"] = py::int_(r.accepted_indices.size());
        row["accepted_indices"] = r.accepted_indices;
        row["accepted_distances"] = r.accepted_distances;
        subject_abc.append(row);
    }

    py::dict estimator;
    estimator["name"] = "ABC";
    estimator["backend"] = "abcpp";
    estimator["method"] = cpp_control.method;
    estimator["reduction"] = cpp_control.reduction;
    py::dict control_used = control_to_dict(cpp_control);
    control_used["n_comp"] = estimator_n_comp(cpp_control, cpp_res);
    estimator["control"] = control_used;

    py::dict diagnostics;
    diagnostics["subject_abc"] = subject_abc;
    diagnostics["summary"] = create_summary_rows(cpp_res);

    py::dict out;
    out["fit"] = fit;
    out["estimator"] = estimator;
    out["diagnostics"] = diagnostics;
    return out;
}

PYBIND11_MODULE(_estimate_abc, m) {
    m.doc() = "metaSDT: estimate_abc wrapper";
    m.def(
        "estimate_abc",
        &py_estimate_abc,
        py::arg("df"),
        py::arg("colnames"),
        py::arg("params"),
        py::arg("model") = "sdt",
        py::arg("control") = py::dict(),
        py::arg("priors") = py::dict()
    );
}
