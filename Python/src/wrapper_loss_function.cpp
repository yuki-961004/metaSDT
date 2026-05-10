#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../Cpp/include/loss_function.hpp"
#include "../../Cpp/include/matrix_mult.hpp"

namespace py = pybind11;

py::dict py_loss_function(
    py::object freq_obj,
    py::object prob_obj,
    py::dict params
) {
    std::vector<std::vector<double>> freq_mat;
    std::vector<std::vector<double>> prob_mat;

    // 智能提取：允许直接传入 matrix_freq 和 matrix_prob 返回的字典
    if (py::isinstance<py::dict>(freq_obj)) {
        freq_mat = freq_obj.cast<py::dict>()["freq_mat"].cast<std::vector<std::vector<double>>>();
    } else {
        freq_mat = freq_obj.cast<std::vector<std::vector<double>>>();
    }
    
    if (py::isinstance<py::dict>(prob_obj)) {
        prob_mat = prob_obj.cast<py::dict>()["prob_mat"].cast<std::vector<std::vector<double>>>();
    } else {
        prob_mat = prob_obj.cast<std::vector<std::vector<double>>>();
    }

    int k = 0;
    if (params.contains("numb_free")) {
        k = params["numb_free"].cast<int>();
    } else {
        throw py::value_error("Error: 'params' must contain 'numb_free'.");
    }
    
    // 提取可能的极小值容差参数 (防止矩阵乘法时 log(0) 报错)
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    if (params.contains("flat")) {
        py::dict flat = params["flat"].cast<py::dict>();
        if (flat.contains("calc_tol")) {
            cpp_params["calc_tol"] = flat["calc_tol"].cast<std::vector<double>>();
        }
    }
    
    // 在 Wrapper 内部先执行矩阵乘法，彻底贴合 Python 用户习惯
    auto cpp_mult = ::matrix_mult(freq_mat, prob_mat, cpp_params);
    LossResult res = loss_function(cpp_mult, freq_mat, k);
    
    py::dict out;
    out["logL"] = res.logL;
    out["nll"] = res.nll;
    out["k"] = res.k;
    out["aic"] = res.aic;
    out["bic"] = res.bic;
    return out;
}

PYBIND11_MODULE(_core_loss_function, m) {
    m.def("loss_function", &py_loss_function, "Calculate model loss (NLL, AIC, BIC).",
          py::arg("freq_mat"), py::arg("prob_mat"), py::arg("params"));
}