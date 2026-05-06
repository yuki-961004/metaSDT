#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Cpp/include/model_sdt.hpp"
#include "Cpp/src/model_sdt.cpp"

namespace py = pybind11;

// 提供一个与 R 语言对标的函数化接口
py::dict py_model_sdt(const std::unordered_map<std::string, std::vector<double>>& params, const std::vector<double>& x) {
    ModelSDT model(params);
    
    py::dict res;
    res["cdf_noise"] = model.cdf_noise(x);
    res["cdf_signal"] = model.cdf_signal(x);
    return res;
}

PYBIND11_MODULE(_model_sdt, m) {
    // 将 C++ 类直接绑定为 Python 类
    py::class_<ModelSDT>(m, "ModelSDT")
        // 绑定构造函数
        .def(py::init<const std::unordered_map<std::string, std::vector<double>>&>())
        
        // 绑定标量版本的方法
        .def("cdf_noise", py::overload_cast<double>(&ModelSDT::cdf_noise, py::const_))
        .def("cdf_signal", py::overload_cast<double>(&ModelSDT::cdf_signal, py::const_))
        // 绑定向量版本的方法 (传入 List/Numpy Array)
        .def("cdf_noise_vec", py::overload_cast<const std::vector<double>&>(&ModelSDT::cdf_noise, py::const_))
        .def("cdf_signal_vec", py::overload_cast<const std::vector<double>&>(&ModelSDT::cdf_signal, py::const_));

    // 导出函数化接口
    m.def("model_sdt", &py_model_sdt, "Evaluate SDT Model CDFs");
}