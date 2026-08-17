#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <metaSDT/model_normal.hpp>

pybind11::dict py_model_normal(pybind11::dict std_params) {
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
    ModelNormal<double> model(cpp_params);
    
    pybind11::dict res;
    res["prob_mat"] = model.compute_probabilities();
    res["criteria"] = model.get_criteria();
    return res;
}

PYBIND11_MODULE(_model_normal, m) {
    pybind11::class_<ModelNormal<double>>(m, "ModelNormal")
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
            return new ModelNormal<double>(cpp_params);
        }))
        .def("compute_probabilities", [](const ModelNormal<double>& self) { 
            return self.compute_probabilities(); 
        })
        .def("get_criteria", [](const ModelNormal<double>& self) {
            return self.get_criteria();
        })
        .def("area", [](const ModelNormal<double>& self,
                        std::size_t stimulus,
                        std::size_t response,
                        double lower,
                        double upper,
                        std::size_t dim_idx) {
            return self.area(stimulus, response, lower, upper, dim_idx);
        });

    m.def("model_normal", &py_model_normal, 
          "Evaluate Normal Meta-Noise Model Probabilities", 
          pybind11::arg("std_params"));
}
