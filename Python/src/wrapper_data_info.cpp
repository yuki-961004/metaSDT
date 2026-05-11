#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../Cpp/include/data_info.hpp"

// 定义一个专门供 Python 端调用的结构体，使其支持点(.)语法访问属性
struct PySubjectData {
    pybind11::dict info;
    pybind11::dict condition;
    pybind11::object raw;
};

pybind11::dict py_data_info(
    pybind11::dict df, 
    pybind11::object colnames = pybind11::none(), 
    pybind11::object condition = pybind11::none()
) {
    // 1. 转换 DataFrame 列数据
    std::unordered_map<std::string, std::vector<double>> cpp_df;
    for (auto item : df) {
        cpp_df[pybind11::str(item.first)] = item.second.cast<std::vector<double>>();
    }

    // 2. 转换列名映射字典
    std::unordered_map<std::string, std::string> cpp_colnames;
    if (!colnames.is_none() && pybind11::isinstance<pybind11::dict>(colnames)) {
        for (auto item : colnames.cast<pybind11::dict>()) {
            cpp_colnames[pybind11::str(item.first)] = pybind11::str(item.second);
        }
    }

    // 3. 转换条件列表
    std::vector<std::string> cpp_condition;
    if (!condition.is_none()) {
        if (pybind11::isinstance<pybind11::str>(condition)) {
            cpp_condition.push_back(condition.cast<std::string>());
        } else if (
            pybind11::isinstance<pybind11::list>(condition) || 
            pybind11::isinstance<pybind11::tuple>(condition)
        ) {
            for (auto item : condition) {
                cpp_condition.push_back(pybind11::str(item));
            }
        }
    }

    // 4. 调用纯 C++ 核心执行极速扫描
    DataInfoResult res;
    try {
        res = data_info(cpp_df, cpp_colnames, cpp_condition);
    } catch (const std::exception& e) {
        throw pybind11::value_error(e.what());
    }

    // 5. 拦截 C++ 产生的警告，并发送给 Python 的 warnings 模块
    pybind11::object warn = pybind11::module_::import("warnings").attr("warn");
    for (const auto& w : res.warnings) {
        warn(pybind11::str(w));
    }

    // 6. 将结果打包回 Python 字典
    pybind11::dict r_subjects;
    for (const auto& kv : res.subjects) {
        double subid = kv.first;
        // 去掉 double 的小数尾巴，比如 1.0 变 "1"
        std::string sub_key = std::to_string(subid);
        sub_key.erase(sub_key.find_last_not_of('0') + 1, std::string::npos);
        if (!sub_key.empty() && sub_key.back() == '.') sub_key.pop_back();

        // 构建我们定义的 PySubjectData 对象
        PySubjectData subj_data;
        subj_data.raw = pybind11::cast(kv.second.raw);

        pybind11::dict r_info;
        r_info["n_trials"] = kv.second.info.n_trials;
        r_info["n_blocks"] = kv.second.info.n_blocks;
        subj_data.info = r_info;

        pybind11::dict py_condition;
        for (const auto& cond_kv : kv.second.condition) {
            // 在底层直接将 "1.0" 或 "1" 清洗并转换为整型键
            try {
                int int_key = static_cast<int>(std::stod(cond_kv.first));
                py_condition[pybind11::int_(int_key)] = pybind11::cast(cond_kv.second);
            } catch (...) {
                // 如果转换失败（比如本来就是纯字符串），则退回使用字符串键
                py_condition[pybind11::str(cond_kv.first)] = pybind11::cast(cond_kv.second);
            }
        }
        subj_data.condition = py_condition;

        r_subjects[pybind11::str(sub_key)] = pybind11::cast(subj_data);
    }

    pybind11::dict out;
    // pybind11::cast 自动处理 unordered_map 到 dict
    out["colnames"] = pybind11::cast(res.colnames); 
    out["subjects"] = r_subjects;
    return out;
}

PYBIND11_MODULE(_core_data_info, m) {
    // 注册 Python 类，使得在 Python 端可以通过点(.)访问
    pybind11::class_<PySubjectData>(m, "SubjectData")
        .def_readonly("info", &PySubjectData::info)
        .def_readonly("condition", &PySubjectData::condition)
        .def_readonly("raw", &PySubjectData::raw);

    m.def("data_info", &py_data_info, 
          "Intelligently scan the dataset and extract subject-level information.",
          pybind11::arg("df"), 
          pybind11::arg("colnames") = pybind11::none(), 
          pybind11::arg("condition") = pybind11::none());
}