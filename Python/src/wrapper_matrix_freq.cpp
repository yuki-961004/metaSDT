#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 自动实现 std::vector 和 python list/dict 的互相转换

#include "../../Cpp/include/matrix_freq.hpp"
#include <unordered_map>

// 辅助函数：将 Python 的各种类数组对象（如 pandas Series, numpy array, list 等）
// 安全地转换为 C++ 的 std::vector<double>
std::vector<double> to_std_vector(pybind11::object obj) {
    // 如果对象有 .tolist() 方法 (比如 pandas Series 或 numpy array)，先调用它
    if (pybind11::hasattr(obj, "tolist")) {
        obj = obj.attr("tolist")();
    }
    // 将标准的 Python 列表(或已被转为列表的对象)安全地映射至 std::vector<double>
    return obj.cast<std::vector<double>>();
}

// Python 接口包装层：负责 Python 对象与 C++ 底层数据结构之间的转换
// 这里将参数改为通用的 pybind11::object，以接收从 Python 侧传来的任意序列对象
pybind11::dict py_matrix_freq(
    pybind11::object stim_obj,
    pybind11::object resp_obj,
    pybind11::object conf_obj = pybind11::none(),
    pybind11::object diff_obj = pybind11::none(),
    pybind11::object std_params_obj = pybind11::none()
) {
    // 1. 数据类型转换与提取
    std::vector<double> stim = to_std_vector(stim_obj);
    std::vector<double> resp = to_std_vector(resp_obj);

    std::vector<double> conf;
    const std::vector<double>* conf_ptr = nullptr;
    
    // 若 Python 端传递了信心指数 (非 None)
    if (!conf_obj.is_none()) {
        conf = to_std_vector(conf_obj);
        conf_ptr = &conf;
    }

    std::vector<double> diff;
    const std::vector<double>* diff_ptr = nullptr;
    if (!diff_obj.is_none()) {
        diff = to_std_vector(diff_obj);
        diff_ptr = &diff;
    }

    std::unordered_map<std::string, std::vector<double>> std_params;
    const std::unordered_map<std::string, std::vector<double>>* params_ptr = nullptr;
    if (!std_params_obj.is_none() && pybind11::isinstance<pybind11::dict>(std_params_obj)) {
        for (auto item : std_params_obj.cast<pybind11::dict>()) {
            std::string key = pybind11::str(item.first);
            if (key == "name_free" || key == "name_fixed" || 
                key == "name_constant" || key == "numb_free" || 
                key == "numb_fixed" || key == "numb_constant" ||
                key == "free_params" || key == "fixed_params" || 
                key == "constant_params") {
                continue;
            }
            std_params[key] = item.second.cast<std::vector<double>>();
        }
        params_ptr = &std_params;
    }

    MatrixFreq res = matrix_freq(
        /*stim=*/stim, /*resp=*/resp, 
        /*conf=*/conf_ptr, /*diff=*/diff_ptr, 
        /*std_params=*/params_ptr
    );

    // 3. 将 C++ 的结构体数据封装为 Python 的字典 (dict) 格式
    // 得益于 <pybind11/stl.h>，C++ 的 std::vector 
    // 及其嵌套数组会被自动且安全地转化为 Python 的 list 列表
    pybind11::dict out;
    out["freq_mat"] = res.freq_mat;
    out["dim_names"] = res.dim_names;
    out["row_names"] = res.row_names;
    out["col_names"] = res.col_names;
    return out;
}

// 4. 注册 Python 原生扩展模块 '_core_matrix_freq' (编译后会在 Python 侧暴露为模块对象)
PYBIND11_MODULE(_core_matrix_freq, m) {
    m.doc() = "metaSDT C++ core backend"; // 设置模块级的文档字符串 (__doc__)
    
    // 5. 绑定接口函数，并为 Python 指定默认参数和关键字参数支持 (kwargs)
    m.def("matrix_freq", &py_matrix_freq, 
          "Calculate the frequency matrix for Signal Detection Theory",
          // 绑定参数名，允许在 Python 端使用关键字参数 
          // (如 matrix_freq(stim=..., resp=...))，并设置 conf 的默认值为 None
          pybind11::arg("stim"), 
          pybind11::arg("resp"), 
          pybind11::arg("conf") = pybind11::none(),
          pybind11::arg("diff") = pybind11::none(),
          pybind11::arg("std_params") = pybind11::none());
}