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
    int k = 0;
    if (params.contains("numb_free")) {
        k = params["numb_free"].cast<int>();
    } else {
        throw py::value_error("Error: 'params' must contain 'numb_free'.");
    }
    
    LossResult res = loss_function(mult_mat, freq_mat, k);
    
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