#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 必须引入：自动处理 C++ STL 和 Python list/dict/str 的无缝双向转换

#include "../../Cpp/include/estimate_mle.hpp"

// 辅助函数：将 Python 嵌套字典提取到 C++ 的 ParamGroup 中
void py_dict_to_cpp_map(
    const pybind11::dict& d, 
    std::unordered_map<std::string, std::vector<double>>& out
) {
    for (auto item : d) {
        std::string key = pybind11::str(item.first);
        out[key] = item.second.cast<std::vector<double>>();
    }
}

// 暴露给 Python 的拟合主函数
pybind11::list py_estimate_mle(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const pybind11::dict& params,
    std::string model = "sdt",
    const pybind11::dict& control = pybind11::dict(),
    const pybind11::dict& lower = pybind11::dict(),
    const pybind11::dict& upper = pybind11::dict()
) {
    // 1. 转换模型参数 (ParamGroup)
    ParamGroup user_params;
    if (params.contains("free")) {
        py_dict_to_cpp_map(params["free"].cast<pybind11::dict>(), user_params.free);
    }
    if (params.contains("fixed")) {
        py_dict_to_cpp_map(params["fixed"].cast<pybind11::dict>(), user_params.fixed);
    }
    if (params.contains("constant")) {
        py_dict_to_cpp_map(params["constant"].cast<pybind11::dict>(), 
                           user_params.constant);
    }
    
    // 支持扁平直接传入
    if (!params.contains("free") && !params.contains("fixed") && 
        !params.contains("constant")) {
        py_dict_to_cpp_map(params, user_params.free);
    }

    // 2. 转换控制参数 (NLoptControl)
    NLoptControl cpp_control;
    if (control.contains("algorithm")) {
        cpp_control.algorithm = control["algorithm"].cast<std::string>();
    }
    if (control.contains("xtol_rel")) {
        cpp_control.xtol_rel = control["xtol_rel"].cast<double>();
    }
    if (control.contains("maxeval")) {
        cpp_control.maxeval = control["maxeval"].cast<int>();
    }
    if (control.contains("ftol_rel")) {
        cpp_control.ftol_rel = control["ftol_rel"].cast<double>();
    }
    if (control.contains("ftol_abs")) {
        cpp_control.ftol_abs = control["ftol_abs"].cast<double>();
    }
    if (control.contains("xtol_abs")) {
        cpp_control.xtol_abs = control["xtol_abs"].cast<double>();
    }
    if (control.contains("maxtime")) {
        cpp_control.maxtime = control["maxtime"].cast<double>();
    }
    if (control.contains("stopval")) {
        cpp_control.stopval = control["stopval"].cast<double>();
    }
    if (control.contains("population")) {
        cpp_control.population = control["population"].cast<int>();
    }
    if (control.contains("initial_step")) {
        cpp_control.initial_step = control["initial_step"].cast<double>();
    }
    if (control.contains("local_algorithm")) {
        cpp_control.local_algorithm = 
            control["local_algorithm"].cast<std::string>();
    }

    // 3. 转换边界参数
    std::unordered_map<std::string, std::vector<double>> cpp_lower, cpp_upper;
    if (lower.size() > 0) py_dict_to_cpp_map(lower, cpp_lower);
    if (upper.size() > 0) py_dict_to_cpp_map(upper, cpp_upper);

    // 4. 调用底层的 C++ 多线程并行 MLE 优化！
    std::vector<SubjectFitResult> cpp_res;
    
    // 【究极大招】：释放 Python GIL 全局解释器锁！
    // 因为我们的底层 C++ 是纯净的，不涉及 Python 对象，
    // 释放 GIL 后，OpenMP 将不受 Python 阻塞，实现真正的全核心并行加速！
    {
        pybind11::gil_scoped_release release; 
        cpp_res = ::estimate_mle(
            df, colnames, user_params, model, cpp_control, cpp_lower, cpp_upper
        );
    } // 离开作用域后自动重新获取 GIL

    // 5. 将结果完美展平为 Python 的 list of dicts (可被 pandas.DataFrame 无缝接收)
    pybind11::list out_list;
    for (const auto& r : cpp_res) {
        pybind11::dict res_dict;
        res_dict["subid"] = r.subid;
        res_dict["logL"] = r.logL;
        res_dict["aic"] = r.aic;
        res_dict["bic"] = r.bic;
        res_dict["status"] = r.status;
        
        // 遍历最佳参数字典，进行清爽的扁平化
        for (const auto& kv : r.best_params) {
            if (kv.second.size() == 1) {
                res_dict[pybind11::str(kv.first)] = kv.second[0]; // 标量参数
            } else {
                // 向量参数 (如 c_conf) 展平为 c_conf_1, c_conf_2 ...
                for (size_t j = 0; j < kv.second.size(); ++j) {
                    std::string key_name = kv.first + "_" + std::to_string(j + 1);
                    res_dict[pybind11::str(key_name)] = kv.second[j];
                }
            }
        }
        out_list.append(res_dict);
    }
    
    return out_list;
}

// ==========================================================
// PYBIND11 模块注册入口
// ==========================================================
PYBIND11_MODULE(_estimate_mle, m) {
    m.doc() = "metaSDT: High-performance Meta-Cognitive SDT modeling via C++";

    // 将 C++ 函数注册给 Python
    m.def("estimate_mle", &py_estimate_mle, 
          "Perform Maximum Likelihood Estimation",
          pybind11::arg("df"),
          pybind11::arg("colnames"),
          pybind11::arg("params"),
          pybind11::arg("model") = "sdt",
          pybind11::arg("control") = pybind11::dict(),
          pybind11::arg("lower") = pybind11::dict(),
          pybind11::arg("upper") = pybind11::dict());
          
    // 注意：pybind11 会自动将 C++ 异常 (std::invalid_argument) 转为 Python 的 ValueError！
}