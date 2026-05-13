#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../Cpp/include/criterion_likelihood.hpp"
#include "../../Cpp/include/matrix_mult.hpp"

pybind11::dict py_criterion_likelihood(
    pybind11::object freq_obj,
    pybind11::object prob_obj,
    pybind11::dict std_params
) {
    std::vector<std::vector<std::vector<double>>> freq_mat;
    std::vector<std::vector<std::vector<double>>> prob_mat;

    if (pybind11::isinstance<pybind11::dict>(freq_obj)) {
        freq_mat = freq_obj.cast<pybind11::dict>()["freq_mat"]
                           .cast<std::vector<std::vector<std::vector<double>>>>();
    } else {
        freq_mat = freq_obj.cast<std::vector<std::vector<std::vector<double>>>>();
    }
    
    if (pybind11::isinstance<pybind11::dict>(prob_obj)) {
        prob_mat = prob_obj.cast<pybind11::dict>()["prob_mat"]
                           .cast<std::vector<std::vector<std::vector<double>>>>();
    } else {
        prob_mat = prob_obj.cast<std::vector<std::vector<std::vector<double>>>>();
    }

    int k = 0;
    if (std_params.contains("numb_free")) {
        k = std_params["numb_free"].cast<int>();
    } else {
        throw pybind11::value_error("Error: 'std_params' must contain 'numb_free'.");
    }
    
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    if (std_params.contains("flat")) {
        pybind11::dict flat = std_params["flat"].cast<pybind11::dict>();
        if (flat.contains("calc_tol")) {
            cpp_params["calc_tol"] = flat["calc_tol"].cast<std::vector<double>>();
        }
    }
    
    auto cpp_mult = ::matrix_mult<double>(freq_mat, prob_mat, cpp_params);
    
    std::vector<double> free_params; 
    auto res = criterion_likelihood<double>(
        cpp_mult, freq_mat, k, free_params, cpp_params
    );
    
    pybind11::dict out;
    out["logL"] = res.logL;
    out["nll"] = res.nll;
    out["k"] = res.k;
    out["aic"] = res.aic;
    out["bic"] = res.bic;
    return out;
}

PYBIND11_MODULE(_core_criterion_likelihood, m) {
    m.def("criterion_likelihood", &py_criterion_likelihood, 
          "Calculate model loss (NLL, AIC, BIC).",
          pybind11::arg("freq_mat"), pybind11::arg("prob_mat"), 
          pybind11::arg("std_params"));
}