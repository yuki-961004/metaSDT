#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 自动实现 std::vector 和 python list/dict 的互相转换
#include <optional>

#include "Cpp/include/matrix_freq.hpp"
// 纯 C++ 源文件已经通过脚本同步至 Python/src/Cpp，继续使用联合编译
#include "Cpp/src/matrix_freq.cpp"

namespace py = pybind11;

// Python 接口包装函数，使用 std::optional 来优雅地处理 None 值
py::dict py_matrix_freq(
    const std::vector<double>& stim,
    const std::vector<double>& resp,
    std::optional<std::vector<double>> conf = std::nullopt
) {
    const std::vector<double>* conf_ptr = nullptr;
    if (conf.has_value()) {
        conf_ptr = &conf.value();
    }

    MatrixFreq res = matrix_freq(stim, resp, conf_ptr);

    // 将 C++ 的 struct 转换为 Python 友好的字典返回
    py::dict out;
    out["freq_mat"] = res.freq_mat;
    out["row_names"] = res.row_names;
    out["col_names"] = res.col_names;
    return out;
}

// 注册 Python 模块 '_core'
PYBIND11_MODULE(_core, m) {
    m.doc() = "metaSDT C++ core backend"; // 模块文档字符串
    m.def("matrix_freq", &py_matrix_freq, 
          "Calculate the frequency matrix for Signal Detection Theory",
          py::arg("stim"), py::arg("resp"), py::arg("conf") = py::none());
}