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
    ModelSDT<double> model(cpp_params);
    
    py::dict res;
    res["cdf_noise"] = model.cdf_noise();
    res["cdf_signal"] = model.cdf_signal();
    return res;
}

PYBIND11_MODULE(_model_sdt, m) {
    // 将 C++ 类直接绑定为 Python 类
    py::class_<ModelSDT<double>>(m, "ModelSDT")
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
            return new ModelSDT<double>(cpp_params);
        }))
        
        // 💡 秘诀：使用 Lambda 表达式完美解决 MSVC 下重载和 const 推导的歧义问题
        .def("cdf_noise", [](const ModelSDT<double>& self) { return self.cdf_noise(); })
        .def("cdf_noise", [](const ModelSDT<double>& self, double x) { return self.cdf_noise(x); })
        .def("cdf_noise_vec", [](const ModelSDT<double>& self, const std::vector<double>& x_vec) { return self.cdf_noise(x_vec); })
        
        .def("cdf_signal", [](const ModelSDT<double>& self) { return self.cdf_signal(); })
        .def("cdf_signal", [](const ModelSDT<double>& self, double x) { return self.cdf_signal(x); })
        .def("cdf_signal_vec", [](const ModelSDT<double>& self, const std::vector<double>& x_vec) { return self.cdf_signal(x_vec); });

    // 导出函数化接口
    m.def("model_sdt", &py_model_sdt, "Evaluate SDT Model CDFs");
}