#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Cpp/include/model_sdt.hpp"
#include "Cpp/src/model_sdt.cpp"

namespace py = pybind11;

// 提供一个与 R 语言对标的函数化接口
py::dict py_model_sdt(py::dict params) {
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
    ModelSDT model(cpp_params);
    
    py::dict res;
    res["cdf_noise"] = model.cdf_noise();
    res["cdf_signal"] = model.cdf_signal();
    return res;
}

PYBIND11_MODULE(_model_sdt, m) {
    // 将 C++ 类直接绑定为 Python 类
    py::class_<ModelSDT>(m, "ModelSDT")
        // 绑定构造函数
        .def(py::init([](py::dict params) {
            std::unordered_map<std::string, std::vector<double>> cpp_params;
            for (auto item : params) {
                std::string key = py::str(item.first);
                if (key == "name_free" || key == "name_fixed" || 
                    key == "name_constant" || key == "numb_free" || 
                    key == "numb_fixed" || key == "numb_constant" ||
                    key == "free_params" || key == "fixed_params" || 
                    key == "constant_params") {
                    continue;
                }
                cpp_params[key] = item.second.cast<std::vector<double>>();
            }
            return new ModelSDT(cpp_params);
        }))
        
        // 绑定无参版本的方法 (自动使用内部 criteria)
        .def("cdf_noise", py::overload_cast<>(&ModelSDT::cdf_noise, py::const_))
        .def("cdf_signal", py::overload_cast<>(&ModelSDT::cdf_signal, py::const_))
        // 绑定标量版本的方法
        .def("cdf_noise", py::overload_cast<double>(&ModelSDT::cdf_noise, py::const_))
        .def("cdf_signal", py::overload_cast<double>(&ModelSDT::cdf_signal, py::const_))
        // 绑定向量版本的方法 (传入 List/Numpy Array)
        .def("cdf_noise_vec", py::overload_cast<const std::vector<double>&>(&ModelSDT::cdf_noise, py::const_))
        .def("cdf_signal_vec", py::overload_cast<const std::vector<double>&>(&ModelSDT::cdf_signal, py::const_));

    // 导出函数化接口
    m.def("model_sdt", &py_model_sdt, "Evaluate SDT Model CDFs");
}