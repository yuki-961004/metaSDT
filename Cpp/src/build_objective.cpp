#include "../include/build_objective.hpp"
#include "../include/data_info.hpp"
#include "../include/matrix_prob.hpp"
#include "../include/matrix_mult.hpp"
#include "../include/model_sdt.hpp"
#include "../include/criterion_likelihood.hpp"

#include <stdexcept>

std::vector<SubjectFitTask> build_fit_tasks(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper
) {
    // 1. 标准化用户输入的模型参数
    ModifiedParamsResult base_params = modify_params(
        user_params, custom_lower, custom_upper
    );

    // 2. 通过 data_info 智能扫描数据集，自动提取被试划分及映射列名
    std::vector<std::string> no_conditions; 
    DataInfoResult info = data_info(df, colnames, no_conditions);

    if (!info.colnames.count("stim") || !info.colnames.count("resp")) {
        throw std::invalid_argument(
            "Error: Missing critical columns 'stim' or 'resp'."
        );
    }

    std::string col_stim = info.colnames.at("stim");
    std::string col_resp = info.colnames.at("resp");
    
    std::string col_conf = "";
    if (info.colnames.count("conf")) {
        col_conf = info.colnames.at("conf");
    }

    // 动态探测：模型参数中是否包含实际的置信度边界
    bool has_conf_params = false;
    if (base_params.flat.count("c_conf") && 
        !base_params.flat.at("c_conf").empty()) {
        has_conf_params = true;
    }

    // 获取原始数据的直接引用。
    // 核心防御：如果模型没有配置 c_conf，即使数据集里有 conf 列，也强制忽略！
    // 否则会导致 Freq 矩阵(多列) 和 Prob 矩阵(2列) 维度不匹配而崩溃。
    const auto& full_stim = df.at(col_stim);
    const auto& full_resp = df.at(col_resp);
    const std::vector<double>* full_conf_ptr = nullptr;
    
    if (has_conf_params) {
        if (!col_conf.empty() && df.count(col_conf)) {
            full_conf_ptr = &df.at(col_conf);
        } else {
            throw std::invalid_argument(
                "Error: Model requires 'c_conf', "
                "but no confidence column found in data."
            );
        }
    }

    std::vector<SubjectFitTask> tasks;
    tasks.reserve(info.subjects.size());

    // 3. 核心解耦：遍历每一个被试，提前计算属于 TA 的频数矩阵
    for (const auto& kv : info.subjects) {
        double subid = kv.first;
        const auto& row_indices = kv.second.raw; // 该被试独有的行索引

        std::vector<double> sub_stim, sub_resp, sub_conf;
        sub_stim.reserve(row_indices.size());
        sub_resp.reserve(row_indices.size());
        if (full_conf_ptr != nullptr) {
            sub_conf.reserve(row_indices.size());
        }

        for (int idx : row_indices) {
            sub_stim.push_back(full_stim[idx]);
            sub_resp.push_back(full_resp[idx]);
            if (full_conf_ptr != nullptr) {
                sub_conf.push_back((*full_conf_ptr)[idx]);
            }
        }

        // 一次性计算出频数矩阵并永久缓存，避免在 NLOPT 循环里计算几千次！
        MatrixFreq freq_obj = matrix_freq(
            sub_stim, sub_resp, full_conf_ptr != nullptr ? &sub_conf : nullptr
        );

        SubjectFitTask task;
        task.subid = subid;
        task.model = model_name;
        task.freq = freq_obj;
        task.params = base_params;
        tasks.push_back(task);
    }

    return tasks;
}

double nll(const std::vector<double>& x, std::vector<double>& grad, void* f_data) {
    // 1. 从虚指针中取回属于该线程、该被试的独立 Context Payload
    SubjectFitTask* task = static_cast<SubjectFitTask*>(f_data);
    
    // 2. 参数动态还原：将 NLOPT 给出的探索值 x，按照结构重组为参数字典
    auto std_params = task->params.flat; // 局部拷贝，线程安全
    size_t x_idx = 0;
    std::vector<double> free_params_vec;
    for (const auto& name : task->params.name_free) {
        // 识别出参数本来的长度并赋值 (比如 c_conf 可能包含多个数值)
        size_t param_len = task->params.structured.free.at(name).size();
        for (size_t i = 0; i < param_len; ++i) {
            double val = x[x_idx++];
            std_params[name][i] = val;
            free_params_vec.push_back(val);
        }
    }

    // 3. 模型正向推导流水线
    std::vector<double> cdf_n;
    std::vector<double> cdf_s;

    // 根据任务包中记录的模型名称，动态选择对应的类进行实例化
    // 未来有新模型（比如 ModelDecay 等），继续往这里加 else if 即可
    if (task->model == "sdt") {
        ModelSDT model(std_params);
        cdf_n = model.cdf_noise();
        cdf_s = model.cdf_signal();
    } else {
        // 如果遇到未知模型直接抛出异常
        throw std::invalid_argument("Error: Unknown model name '" + task->model + "'.");
    }

    MatrixProb<double> prob = matrix_prob<double>(cdf_n, cdf_s, std_params);

    // 注意：task->freq.freq_mat 是在 builder 里就缓存好的，计算极快
    auto mult = matrix_mult<double>(task->freq.freq_mat, prob.prob_mat, std_params);

    auto loss = criterion_likelihood(
        mult, task->freq.freq_mat, task->params.numb_free, 
        free_params_vec, std_params
    );

    // 4. 返回负对数似然用于最小化
    return loss.nll;
}