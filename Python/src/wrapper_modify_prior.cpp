#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../Cpp/include/modify_params.hpp"
#include "../../Cpp/include/modify_prior.hpp"

namespace {
    std::vector<double> extract_to_vector(pybind11::object val) {
        if (pybind11::hasattr(val, "tolist")) val = val.attr("tolist")();
        if (pybind11::isinstance<pybind11::sequence>(val) && 
            !pybind11::isinstance<pybind11::str>(val)) {
            return val.cast<std::vector<double>>();
        }
        return {val.cast<double>()};
    }

    void py_dict_to_cpp_map(
        pybind11::object py_obj, 
        std::unordered_map<std::string, std::vector<double>>& cpp_map
    ) {
        if (py_obj.is_none()) return;
        if (!pybind11::isinstance<pybind11::dict>(py_obj)) {
            throw pybind11::type_error("Params must be a dictionary.");
        }
        pybind11::dict d = py_obj.cast<pybind11::dict>();
        for (auto item : d) {
            std::string key = pybind11::str(item.first);
            cpp_map[key] = extract_to_vector(
                pybind11::reinterpret_borrow<pybind11::object>(item.second)
            );
        }
    }
}

pybind11::dict py_modify_prior(
    pybind11::dict user_priors, 
    pybind11::object std_params = pybind11::none()
) {
    // 1. 解析 Params
    ModifiedParamsResult param_info;
    if (!std_params.is_none()) {
        pybind11::dict d = std_params.cast<pybind11::dict>();
        if (d.contains("name_free")) {
            param_info.name_free = d["name_free"].cast<std::vector<std::string>>();
            if (d.contains("name_fixed")) param_info.name_fixed = d["name_fixed"].cast<std::vector<std::string>>();
            if (d.contains("name_constant")) param_info.name_constant = d["name_constant"].cast<std::vector<std::string>>();
            for (const auto& name : param_info.name_free) param_info.structured.free[name] = d[name.c_str()].cast<std::vector<double>>();
            for (const auto& name : param_info.name_fixed) param_info.structured.fixed[name] = d[name.c_str()].cast<std::vector<double>>();
            for (const auto& name : param_info.name_constant) param_info.structured.constant[name] = d[name.c_str()].cast<std::vector<double>>();
        } else {
            ParamGroup cpp_user_params;
            bool is_strc = d.contains("free") || d.contains("fixed") || d.contains("constant");
            if (is_strc) {
                if (d.contains("free")) py_dict_to_cpp_map(d["free"], cpp_user_params.free);
                if (d.contains("fixed")) py_dict_to_cpp_map(d["fixed"], cpp_user_params.fixed);
                if (d.contains("constant")) py_dict_to_cpp_map(d["constant"], cpp_user_params.constant);
            } else {
                py_dict_to_cpp_map(std_params, cpp_user_params.free);
            }
            param_info = modify_params(cpp_user_params);
        }
    } else {
        ParamGroup cpp_user_params;
        param_info = modify_params(cpp_user_params);
    }

    // 2. 解析 Priors
    std::unordered_map<std::string, UserPrior> cpp_user_priors;
    for (auto item : user_priors) {
        std::string p_name = pybind11::str(item.first);
        pybind11::dict spec = item.second.cast<pybind11::dict>();
        UserPrior up;
        if (!spec.contains("type")) {
            throw pybind11::value_error("Prior spec must contain 'type'.");
        }
        up.type = spec["type"].cast<std::string>();
        for (auto arg_item : spec) {
            std::string key = pybind11::str(arg_item.first);
            if (key != "type") up.args[key] = arg_item.second.cast<double>();
        }
        cpp_user_priors[p_name] = up;
    }

    // 3. 构建核心优先引擎
    CriterionPrior cp = modify_prior(cpp_user_priors, param_info);

    // 4. 返回打包结构
    pybind11::dict out;
    for (const auto& kv : cp.prior_specs_) {
        pybind11::dict p_dict;
        std::string type_str;
        switch(kv.second.type) {
            case CriterionPrior::PriorType::NORMAL: type_str = "normal"; break;
            case CriterionPrior::PriorType::UNIFORM: type_str = "uniform"; break;
            case CriterionPrior::PriorType::LOGNORMAL: type_str = "lognormal"; break;
            case CriterionPrior::PriorType::CAUCHY: type_str = "cauchy"; break;
            case CriterionPrior::PriorType::BETA: type_str = "beta"; break;
            case CriterionPrior::PriorType::EXPONENTIAL: type_str = "exponential"; break;
            case CriterionPrior::PriorType::NONE: type_str = "none"; break;
        }
        p_dict["type"] = type_str;
        p_dict["param1"] = kv.second.param1;
        p_dict["param2"] = kv.second.param2;
        out[pybind11::int_(kv.first)] = p_dict;
    }
    return out;
}

PYBIND11_MODULE(_help_modify_prior, m) {
    m.def("modify_prior", &py_modify_prior,
          "Modify and align prior distributions to flattened parameters", 
          pybind11::arg("user_priors"), 
          pybind11::arg("std_params") = pybind11::none());
}