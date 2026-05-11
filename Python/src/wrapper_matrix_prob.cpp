#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../Cpp/include/matrix_prob.hpp"

pybind11::dict py_matrix_prob(
    const std::vector<double>& cdf_noise,
    const std::vector<double>& cdf_signal,
    pybind11::dict std_params
) {
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
    // 调用底层的 C++ 函数计算概率矩阵
    MatrixProb<double> mat = matrix_prob<double>(cdf_noise, cdf_signal, cpp_params);
    
    // 封装为 Python 的字典，方便外层套壳转换为 Pandas DataFrame
    pybind11::dict res;
    res["prob_mat"] = mat.prob_mat;
    res["row_names"] = mat.row_names;
    res["col_names"] = mat.col_names;
    return res;
}

PYBIND11_MODULE(_core_matrix_prob, m) {
    m.def("matrix_prob", &py_matrix_prob, 
          "Generate theoretical probability matrix for SDT",
          pybind11::arg("cdf_noise"), pybind11::arg("cdf_signal"), pybind11::arg("std_params"));
}