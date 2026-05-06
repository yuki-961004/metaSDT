#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Cpp/include/matrix_prob.hpp"
#include "Cpp/src/matrix_prob.cpp"

namespace py = pybind11;

py::dict py_matrix_prob(
    const std::vector<double>& cdf_noise,
    const std::vector<double>& cdf_signal,
    const std::unordered_map<std::string, std::vector<double>>& params
) {
    // 调用底层的 C++ 函数计算概率矩阵
    MatrixProb mat = matrix_prob(cdf_noise, cdf_signal, params);
    
    // 封装为 Python 的字典，方便外层套壳转换为 Pandas DataFrame
    py::dict res;
    res["prob_mat"] = mat.prob_mat;
    res["row_names"] = mat.row_names;
    res["col_names"] = mat.col_names;
    return res;
}

PYBIND11_MODULE(_core_matrix_prob, m) {
    m.def("matrix_prob", &py_matrix_prob, "Generate theoretical probability matrix for SDT");
}