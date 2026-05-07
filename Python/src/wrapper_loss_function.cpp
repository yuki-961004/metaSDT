#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Cpp/include/loss_function.hpp"
#include "Cpp/src/loss_function.cpp"

namespace py = pybind11;

py::dict py_loss_function(
    const std::vector<std::vector<double>>& mult_mat,
    const std::vector<std::vector<double>>& freq_mat,
    py::dict params
) {
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    std::vector<std::string> free_params;

    for (auto item : params) {
        std::string key = py::str(item.first);
        if (key == "free_params") {
            free_params = item.second.cast<std::vector<std::string>>();
            continue;
        }
        if (key == "fixed_params" || key == "constant_params") continue;
        cpp_params[key] = item.second.cast<std::vector<double>>();
    }
    
    LossResult res = loss_function(mult_mat, freq_mat, cpp_params, free_params);
    
    py::dict out;
    out["logL"] = res.logL;
    out["nll"] = res.nll;
    out["k"] = res.k;
    out["aic"] = res.aic;
    out["bic"] = res.bic;
    return out;
}

PYBIND11_MODULE(_core_loss_function, m) {
    m.def("loss_function", &py_loss_function, "Calculate model loss (NLL, AIC, BIC).");
}