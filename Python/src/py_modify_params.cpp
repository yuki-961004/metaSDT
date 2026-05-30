#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 自动实现 std::vector, std::unordered_map �?Python dict/list 的互相转�?

#include <metaSDT/modify_params.hpp>
#include <metaSDT/modify_priors.hpp>

// 辅助函数 1: �?Python 端的单个�?标量或序�?安全转换�?C++ �?std::vector<double>
std::vector<double> extract_to_vector(pybind11::object val) {
    // 如果�?numpy array �?pandas Series，先调用 tolist() 转换为原生列�?
    if (pybind11::hasattr(val, "tolist")) {
        val = val.attr("tolist")();
    }
    
    // 如果是序�?比如 list, tuple), 但不能是字符�? 直接安全�?cast �?vector
    if (
        pybind11::isinstance<pybind11::sequence>(val) && 
        !pybind11::isinstance<pybind11::str>(val)
    ) {
        return val.cast<std::vector<double>>();
    } else {
        // 如果是标�?比如 int, float), 将其包裹进长度为 1 �?vector �?
        return {val.cast<double>()};
    }
}

// 辅助函数 2: �?Python 字典转换�?C++ �?unordered_map<string, vector<double>>
void py_dict_to_cpp_map(
    pybind11::object py_obj, 
    std::unordered_map<std::string, std::vector<double>>& cpp_map
) {
    if (py_obj.is_none()) return; // None 等同于为�? 直接跳过

    if (!pybind11::isinstance<pybind11::dict>(py_obj)) {
        throw pybind11::type_error(
            "Parameters must be provided as a dictionary (dict)."
        );
    }

    pybind11::dict d = py_obj.cast<pybind11::dict>();
    for (auto item : d) {
        // 提取键名�?std::string
        std::string key = pybind11::str(item.first);
        // item.second �?pybind11::handle,
        // 需转为 pybind11::object 后交�?extract_to_vector 解析数�?
        cpp_map[key] = extract_to_vector(
            pybind11::reinterpret_borrow<pybind11::object>(item.second)
        );
    }
}

// Python 接口包装�? 负责 Python �?dict �?C++ �?ParamGroup 之间的智能转�?
pybind11::dict py_modify_params(pybind11::object user_params = pybind11::none()) {
    ParamGroup cpp_user_params;

    if (!user_params.is_none()) {
        if (!pybind11::isinstance<pybind11::dict>(user_params)) {
            throw pybind11::type_error(
                "Input 'params' must be a dictionary or None."
            );
        }

        pybind11::dict d = user_params.cast<pybind11::dict>();
        
        // 判断输入是否是带�?free/fixed/constant 层级的结构化字典
        bool is_structured = d.contains("free") || 
                             d.contains("fixed") || 
                             d.contains("constant");

        if (is_structured) {
            if (d.contains("free")) {
                py_dict_to_cpp_map(
                    /*py_obj=*/d["free"], /*cpp_map=*/cpp_user_params.free
                );
            }
            if (d.contains("fixed")) {
                py_dict_to_cpp_map(
                    /*py_obj=*/d["fixed"], /*cpp_map=*/cpp_user_params.fixed
                );
            }
            if (d.contains("constant")) {
                py_dict_to_cpp_map(
                    /*py_obj=*/d["constant"], /*cpp_map=*/cpp_user_params.constant
                );
            }
        } else {
            // 如果是一维扁平字�?(�?{"d": 2.5}), 默认将其所有元素视�?free 参数
            py_dict_to_cpp_map(
                /*py_obj=*/user_params, /*cpp_map=*/cpp_user_params.free
            );
        }
    }

    // 调用�?C++ 底层的核心计算函�?
    auto cpp_result = modify_params(/*user_params=*/cpp_user_params);

    // 得益�?<pybind11/stl.h>, C++ �?map 会被自动且安全地转化�?Python �?dict
    pybind11::dict out_dict = pybind11::cast(cpp_result.flat);
    
    out_dict["name_free"] = cpp_result.name_free;
    out_dict["name_fixed"] = cpp_result.name_fixed;
    out_dict["name_constant"] = cpp_result.name_constant;
    
    out_dict["lower_bounds"] = cpp_result.lower_bounds;
    out_dict["upper_bounds"] = cpp_result.upper_bounds;
    
    out_dict["numb_free"] = cpp_result.numb_free;
    out_dict["numb_fixed"] = cpp_result.numb_fixed;
    out_dict["numb_constant"] = cpp_result.numb_constant;
    
    return out_dict;
}

PYBIND11_MODULE(_help_modify_params, m) {
    m.def("modify_params", &py_modify_params,
          "Modify and flatten model parameters", 
          pybind11::arg("user_params") = pybind11::none());
}