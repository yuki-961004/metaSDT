#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <metaSDT/model_lognormal.hpp>

pybind11::dict py_model_lognormal(pybind11::dict std_params) {
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
    ModelLognormal<double> model(cpp_params);
    
    pybind11::dict res;
    res["prob_mat"] = model.compute_probabilities();
    res["criteria"] = model.get_criteria();
    return res;
}

PYBIND11_MODULE(_model_lognormal, m) {
    pybind11::class_<ModelLognormal<double>>(m, "ModelLognormal")
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
            return new ModelLognormal<double>(cpp_params);
        }))
        .def("compute_probabilities", [](const ModelLognormal<double>& self) { 
            return self.compute_probabilities(); 
        })
        .def("get_criteria", [](const ModelLognormal<double>& self) {
            return self.get_criteria();
        })
        .def("area", [](const ModelLognormal<double>& self,
                        std::size_t stimulus,
                        std::size_t response,
                        double lower,
                        double upper,
                        std::size_t dim_idx) {
            return self.area(stimulus, response, lower, upper, dim_idx);
        });

    m.def("model_lognormal", &py_model_lognormal, 
          "Evaluate Log-normal Meta-Noise Model Probabilities", 
          pybind11::arg("std_params"));
}
