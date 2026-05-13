#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

#include "../../Cpp/include/modify_params.hpp"
#include "../../Cpp/include/modify_prior.hpp"
#include "../../Cpp/include/criterion_posterior.hpp"

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
        pybind11::dict d = py_obj.cast<pybind11::dict>();
        for (auto item : d) {
            std::string key = pybind11::str(item.first);
            cpp_map[key] = extract_to_vector(
                pybind11::reinterpret_borrow<pybind11::object>(item.second)
            );
        }
    }
}

double py_criterion_posterior(
    pybind11::object freq_obj,
    pybind11::dict user_priors,
    pybind11::object std_params = pybind11::none()
) {
    std::vector<std::vector<std::vector<double>>> freq_mat;
    if (pybind11::isinstance<pybind11::dict>(freq_obj)) {
        freq_mat = freq_obj.cast<pybind11::dict>()["freq_mat"]
                           .cast<std::vector<std::vector<std::vector<double>>>>();
    } else {
        freq_mat = freq_obj.cast<std::vector<std::vector<std::vector<double>>>>();
    }

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
            ParamGroup user_params;
            bool is_strc = d.contains("free") || d.contains("fixed") || d.contains("constant");
            if (is_strc) {
                if (d.contains("free")) py_dict_to_cpp_map(d["free"], user_params.free);
                if (d.contains("fixed")) py_dict_to_cpp_map(d["fixed"], user_params.fixed);
                if (d.contains("constant")) py_dict_to_cpp_map(d["constant"], user_params.constant);
            } else {
                py_dict_to_cpp_map(std_params, user_params.free);
            }
            param_info = modify_params(user_params);
        }
    } else {
        ParamGroup user_params;
        param_info = modify_params(user_params);
    }

    std::unordered_map<std::string, UserPrior> cpp_user_priors;
    for (auto item : user_priors) {
        std::string p_name = pybind11::str(item.first);
        pybind11::dict spec = item.second.cast<pybind11::dict>();
        UserPrior up;
        up.type = spec["type"].cast<std::string>();
        for (auto arg_item : spec) {
            std::string key = pybind11::str(arg_item.first);
            if (key != "type") up.args[key] = arg_item.second.cast<double>();
        }
        cpp_user_priors[p_name] = up;
    }
    CriterionPrior cp = modify_prior(cpp_user_priors, param_info);

    std::vector<int> param_sizes;
    for (const auto& name : param_info.name_free) {
        param_sizes.push_back(param_info.structured.free.at(name).size());
    }
    
    std::unordered_map<std::string, std::vector<double>> static_params;
    for (const auto& kv : param_info.structured.fixed) static_params[kv.first] = kv.second;
    for (const auto& kv : param_info.structured.constant) static_params[kv.first] = kv.second;

    CriterionPosterior posterior(freq_mat, param_info.name_free, param_sizes, static_params, cp);
    
    // 自动在 Wrapper 层组装用于底层计算的扁平探索数组
    std::vector<double> cpp_free_params;
    for (const auto& name : param_info.name_free) {
        const auto& vals = param_info.structured.free.at(name);
        for (double v : vals) cpp_free_params.push_back(v);
    }
    
    Eigen::Map<const Eigen::VectorXd> free_params_eigen(cpp_free_params.data(), cpp_free_params.size());
    return posterior.operator()<double>(free_params_eigen);
}

PYBIND11_MODULE(_core_criterion_posterior, m) {
    m.def("criterion_posterior", &py_criterion_posterior, 
          "Evaluate Unnormalized Log-Posterior",
          pybind11::arg("freq_mat"), 
          pybind11::arg("user_priors"), pybind11::arg("std_params") = pybind11::none());
}