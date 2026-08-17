#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <metaSDT/model_bch.hpp>

pybind11::dict py_model_bch(pybind11::dict std_params) {
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    for (auto item : std_params) {
        std::string key = pybind11::str(item.first);
        if (key == "name_free" || key == "name_fixed" || 
            key == "name_constant" || key == "numb_free" || 
            key == "numb_fixed" || key == "numb_constant" ||
            key == "free_params" || key == "fixed_params" || 
            key == "constant_params") {
            continue;
        }
        cpp_params[key] = item.second.cast<std::vector<double>>();
    }
    ModelBCH<double> model(cpp_params);
    
    pybind11::dict res;
    res["cdf_noise"] = model.cdf_noise();
    res["cdf_signal"] = model.cdf_signal();
    res["criteria"] = model.get_criteria_matrix();
    res["p_thresholds"] = model.get_p_thresholds();
    return res;
}

PYBIND11_MODULE(_model_bch, m) {
    pybind11::class_<ModelBCH<double>>(m, "ModelBCH")
        .def(pybind11::init([](pybind11::dict std_params) {
            std::unordered_map<std::string, std::vector<double>> cpp_params;
            for (auto item : std_params) {
                std::string key = pybind11::str(item.first);
                if (key == "name_free" || key == "name_fixed" || 
                    key == "name_constant" || key == "numb_free" || 
                    key == "numb_fixed" || key == "numb_constant" ||
                    key == "free_params" || key == "fixed_params" || 
                    key == "constant_params") {
                    continue;
                }
                cpp_params[key] = item.second.cast<std::vector<double>>();
            }
            return new ModelBCH<double>(cpp_params);
        }))
        .def("cdf_noise", [](const ModelBCH<double>& self) { 
            return self.cdf_noise(); 
        })
        .def("cdf_signal", [](const ModelBCH<double>& self) { 
            return self.cdf_signal(); 
        })
        .def("get_criteria", [](const ModelBCH<double>& self) {
            return self.get_criteria_matrix();
        })
        .def("get_p_thresholds", [](const ModelBCH<double>& self) {
            return self.get_p_thresholds();
        })
        .def("area", [](const ModelBCH<double>& self,
                        std::size_t stimulus,
                        std::size_t response,
                        double lower,
                        double upper,
                        std::size_t dim_idx) {
            return self.area(stimulus, response, lower, upper, dim_idx);
        });

    m.def("model_bch", &py_model_bch, 
          "Evaluate BCH Model CDFs", 
          pybind11::arg("std_params"));
}
