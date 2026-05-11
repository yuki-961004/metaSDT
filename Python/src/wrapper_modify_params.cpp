#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 自动实现 std::vector, std::unordered_map 与 Python dict/list 的互相转换

#include "../../Cpp/include/modify_params.hpp"
#include "../../Cpp/include/modify_prior.hpp"

// 辅助函数 1：将 Python 端的单个值（标量或序列）安全转换为 C++ 的 std::vector<double>
std::vector<double> extract_to_vector(pybind11::object val) {
    // 如果是 numpy array 或 pandas Series，先调用 tolist() 转换为原生列表
    if (pybind11::hasattr(val, "tolist")) {
        val = val.attr("tolist")();
    }
    
    // 如果是序列（比如 list, tuple），但不能是字符串，直接安全地 cast 为 vector
    if (
        pybind11::isinstance<pybind11::sequence>(val) && 
        !pybind11::isinstance<pybind11::str>(val)
    ) {
        return val.cast<std::vector<double>>();
    } else {
        // 如果是标量（比如 int, float），将其包裹进长度为 1 的 vector 中
        return {val.cast<double>()};
    }
}

// 辅助函数 2：将 Python 字典转换为 C++ 的 unordered_map<string, vector<double>>
void py_dict_to_cpp_map(
    pybind11::object py_obj, 
    std::unordered_map<std::string, std::vector<double>>& cpp_map
) {
    if (py_obj.is_none()) return; // None 等同于为空，直接跳过

    if (!pybind11::isinstance<pybind11::dict>(py_obj)) {
        throw pybind11::type_error(
            "Parameters must be provided as a dictionary (dict)."
        );
    }

    pybind11::dict d = py_obj.cast<pybind11::dict>();
    for (auto item : d) {
        // 提取键名为 std::string
        std::string key = pybind11::str(item.first);
        // item.second 是 pybind11::handle，
        // 需转为 pybind11::object 后交给 extract_to_vector 解析数值
        cpp_map[key] = extract_to_vector(
            pybind11::reinterpret_borrow<pybind11::object>(item.second)
        );
    }
}

// Python 接口包装层：负责 Python 的 dict 与 C++ 的 ParamGroup 之间的智能转换
pybind11::dict py_modify_params(pybind11::object user_params = pybind11::none()) {
    ParamGroup cpp_user_params;

    if (!user_params.is_none()) {
        if (!pybind11::isinstance<pybind11::dict>(user_params)) {
            throw pybind11::type_error(
                "Input 'params' must be a dictionary or None."
            );
        }

        pybind11::dict d = user_params.cast<pybind11::dict>();
        
        // 判断输入是否是带有 free/fixed/constant 层级的结构化字典
        bool is_structured = d.contains("free") || 
                             d.contains("fixed") || 
                             d.contains("constant");

        if (is_structured) {
            if (d.contains("free")) {
                py_dict_to_cpp_map(d["free"], cpp_user_params.free);
            }
            if (d.contains("fixed")) {
                py_dict_to_cpp_map(d["fixed"], cpp_user_params.fixed);
            }
            if (d.contains("constant")) {
                py_dict_to_cpp_map(d["constant"], cpp_user_params.constant);
            }
        } else {
            // 如果是一维扁平字典 (如 {"d": 2.5})，默认将其所有元素视作 free 参数
            py_dict_to_cpp_map(user_params, cpp_user_params.free);
        }
    }

    // 调用纯 C++ 底层的核心计算函数
    auto cpp_result = modify_params(cpp_user_params);

    // 得益于 <pybind11/stl.h>，C++ 的 map 会被自动且安全地转化为 Python 的 dict
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