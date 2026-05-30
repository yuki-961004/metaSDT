#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>
#include <vector>

#include <metaSDT/shell_run_m.hpp>

namespace py = pybind11;

namespace {

/* ========================================================================== *
 *                              Python Parsing                                *
 * ========================================================================== */

std::vector<double> extract_to_vector(py::object value) {
    if (py::hasattr(value, "tolist")) {
        value = value.attr("tolist")();
    }

    if (py::isinstance<py::sequence>(value) &&
        !py::isinstance<py::str>(value) &&
        !py::isinstance<py::bytes>(value)) {
        return value.cast<std::vector<double>>();
    }

    return {value.cast<double>()};
}

void py_dict_to_cpp_map(
    py::object value,
    std::unordered_map<std::string, std::vector<double>>& out
) {
    if (value.is_none()) {
        return;
    }
    if (!py::isinstance<py::dict>(value)) {
        throw py::type_error("Parameter slots must be dictionaries.");
    }

    py::dict dict_value = value.cast<py::dict>();
    for (auto item : dict_value) {
        const std::string key = py::str(item.first);
        py::object item_value = py::reinterpret_borrow<py::object>(
            item.second
        );
        out[key] = extract_to_vector(item_value);
    }
}

ParamGroup params_to_cpp(py::object params) {
    ParamGroup out;
    if (params.is_none()) {
        return out;
    }
    if (!py::isinstance<py::dict>(params)) {
        throw py::type_error("params must be a dictionary or None.");
    }

    py::dict params_dict = params.cast<py::dict>();
    const bool structured = params_dict.contains("free") ||
        params_dict.contains("fixed") ||
        params_dict.contains("constant");

    if (structured) {
        if (params_dict.contains("free")) {
            py_dict_to_cpp_map(params_dict["free"], out.free);
        }
        if (params_dict.contains("fixed")) {
            py_dict_to_cpp_map(params_dict["fixed"], out.fixed);
        }
        if (params_dict.contains("constant")) {
            py_dict_to_cpp_map(params_dict["constant"], out.constant);
        }
    } else {
        py_dict_to_cpp_map(params, out.free);
    }

    return out;
}

ShellRunMOptions option_to_cpp(py::object option) {
    ShellRunMOptions out;
    if (option.is_none()) {
        return out;
    }
    if (!py::isinstance<py::dict>(option)) {
        throw py::type_error("option must be a dictionary or None.");
    }

    py::dict opt = option.cast<py::dict>();
    if (opt.contains("n") && !opt["n"].is_none()) {
        out.n = opt["n"].cast<int>();
    }
    if (opt.contains("seed") && !opt["seed"].is_none()) {
        out.seed = opt["seed"].cast<int>();
        out.has_seed = true;
    }
    if (opt.contains("density_points") && !opt["density_points"].is_none()) {
        out.density_points = opt["density_points"].cast<int>();
    }
    if (opt.contains("xlim") && !opt["xlim"].is_none()) {
        out.xlim = opt["xlim"].cast<std::vector<double>>();
        out.has_xlim = true;
    }

    return out;
}

py::dict data_to_dict(const ShellRunMData& data) {
    py::dict out;
    out["trial"] = data.trial;
    out["stim"] = data.stim;
    out["resp"] = data.resp;
    out["conf"] = data.conf;
    out["diff"] = data.diff;
    out["evidence"] = data.evidence;
    return out;
}

py::dict density_to_dict(const ShellRunMDensity& density) {
    py::dict out;
    out["x"] = density.x;
    out["noise"] = density.noise;
    out["signal"] = density.signal;
    return out;
}

} // namespace

/* ========================================================================== *
 *                              Module Export                                 *
 * ========================================================================== */

py::dict py_shell_run_m(
    py::object params,
    const std::string& model,
    py::object option
) {
    const ParamGroup cpp_params = params_to_cpp(params);
    const ShellRunMOptions cpp_option = option_to_cpp(option);
    const ShellRunMResult result = shell_run_m(cpp_params, model, cpp_option);

    py::dict out;
    out["params"] = result.params;
    out["model"] = result.model;
    out["data"] = data_to_dict(result.data);
    out["density"] = density_to_dict(result.density);
    out["criteria"] = result.criteria;
    out["seed"] = result.seed;
    return out;
}

PYBIND11_MODULE(_shell_run_m, m) {
    m.doc() = "metaSDT: shell_run_m simulation wrapper";
    m.def(
        "shell_run_m",
        &py_shell_run_m,
        "Simulate SDT trial data without fitting",
        py::arg("params"),
        py::arg("model") = "sdt",
        py::arg("option") = py::none()
    );
}
