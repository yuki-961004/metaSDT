#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Cpp/include/matrix_mult.hpp"
#include "Cpp/src/matrix_mult.cpp"

namespace py = pybind11;

std::vector<std::vector<double>> py_matrix_mult(
    const std::vector<std::vector<double>>& freq_mat,
    const std::vector<std::vector<double>>& prob_mat,
    py::dict params
) {
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
          "Calculate the Log-Likelihood product matrix between frequency and probability matrices.");
}