#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../Cpp/include/matrix_mult.hpp"

std::vector<std::vector<std::vector<double>>> py_matrix_mult(
    pybind11::object freq_obj,
    pybind11::object prob_obj,
    pybind11::dict std_params
) {
    std::vector<std::vector<std::vector<double>>> freq_mat;
    std::vector<std::vector<std::vector<double>>> prob_mat;

    // 智能提取：如果传来的是包装好的字典，自动提取其中的矩阵
    if (pybind11::isinstance<pybind11::dict>(freq_obj)) {
        freq_mat = freq_obj.cast<pybind11::dict>()["freq_mat"]
            .cast<std::vector<std::vector<std::vector<double>>>>();
    } else {
        freq_mat = freq_obj
            .cast<std::vector<std::vector<std::vector<double>>>>();
    }
    
    if (pybind11::isinstance<pybind11::dict>(prob_obj)) {
        prob_mat = prob_obj.cast<pybind11::dict>()["prob_mat"]
            .cast<std::vector<std::vector<std::vector<double>>>>();
    } else {
        prob_mat = prob_obj
            .cast<std::vector<std::vector<std::vector<double>>>>();
    }

    std::unordered_map<std::string, std::vector<double>> cpp_params;
    for (auto item : std_params) {
        std::string key = pybind11::str(item.first);
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
    return matrix_mult<double>(
        /*freq_mat=*/freq_mat, /*prob_mat=*/prob_mat, /*std_params=*/cpp_params
    );
}

PYBIND11_MODULE(_core_matrix_mult, m) {
    m.def("matrix_mult", &py_matrix_mult,
          "Calculate the Log-Likelihood product matrix "
          "between frequency and probability matrices.",
          pybind11::arg("freq_mat"), pybind11::arg("prob_mat"), 
          pybind11::arg("std_params"));
}