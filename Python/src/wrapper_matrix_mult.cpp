#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Cpp/include/matrix_mult.hpp"
#include "Cpp/src/matrix_mult.cpp"

namespace py = pybind11;

std::vector<std::vector<double>> py_matrix_mult(
    py::object freq_obj,
    py::object prob_obj,
    py::dict params
) {
    std::vector<std::vector<double>> freq_mat;
    std::vector<std::vector<double>> prob_mat;

    // 智能提取：如果传来的是包装好的字典，自动提取其中的矩阵
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

    std::unordered_map<std::string, std::vector<double>> cpp_params;
    for (auto item : params) {
        std::string key = py::str(item.first);
        // 拦截包含元数据的槽位
        if (key == "name_free" || key == "name_fixed" || 
            key == "name_constant" || key == "numb_free" || 
            key == "numb_fixed" || key == "numb_constant" ||
            key == "free_params" || key == "fixed_params" || 
            key == "constant_params") {
            continue;
        }
        cpp_params[key] = item.second.cast<std::vector<double>>();
    }
    
    // 调用底层的 C++ 函数计算
    return matrix_mult(freq_mat, prob_mat, cpp_params);
}

PYBIND11_MODULE(_core_matrix_mult, m) {
    m.def("matrix_mult", &py_matrix_mult, 
          "Calculate the Log-Likelihood product matrix between frequency and probability matrices.",
          py::arg("freq_mat"), py::arg("prob_mat"), py::arg("params"));
}