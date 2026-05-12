#include "../include/build_objective.hpp"
#include "../include/data_info.hpp"
#include "../include/matrix_prob.hpp"
#include "../include/matrix_mult.hpp"
#include "../include/model_sdt.hpp"
#include "../include/criterion_likelihood.hpp"

#include <stdexcept>
#include <iostream>
#include <set>

// ////////////////////////////////////////////////////////////////////////////
// 概念映射 (给 R 用户的快速指南):
// 1. 在 R 里，你通常会用 foreach 切分数据，并在循环里创建局部的 env 
//    把数据 (如 freq) 和固定的模型参数存进去。
// 2. 在 C++ 里，没有隐式的环境 (env)，取而代之的是显式的“结构体” 
//    即 SubjectFitTask。这个工厂函数就是为了提前给每一个被试、每一条件
//    打包好他们专属的 SubjectFitTask (也就是你的专属 local env)。
// 3. 之后 OpenMP (多线程机制) 会并行地把这些打包好的 Task 喂给 NLOPT 
//    NLOPT 在跑的时候，每一步都会把这个 Task 当做 void 指针 (f_data) 
//    传回给目标函数 nll。这就实现了 R 语言里的局部变量读取！
// ////////////////////////////////////////////////////////////////////////////

std::vector<SubjectFitTask> build_fit_tasks(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper
) {
    // ////////////////////////////////////////////////////////////////////////
    // 1. 数据智能扫描 (Data Scanning & Splitting)
    // 这就好比你在 R 里用 split(df, df$subid) 提前把被试数据切分好。
    // 同时底层会利用正则匹配，自动找出 stim, resp, conf 等关键列的真实名称。
    // ////////////////////////////////////////////////////////////////////////
    DataInfoResult info = data_info(/*df=*/df, /*colnames=*/colnames);

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
    
    std::string col_diff = "";
    if (info.colnames.count("diff")) {
        col_diff = info.colnames.at("diff");
    }

    // ////////////////////////////////////////////////////////////////////////
    // 2. 动态维度防御 (Dynamic Dimension Defense)
    // 探测模型是否含有置信度边界 (c_conf)。如果有，则说明需要置信度数据。
    // 这是一个关键的“安检门”，强制让数据列和模型参数严丝合缝，
    // 彻底杜绝 Freq 矩阵 (多列) 与 Prob 矩阵 (2列) 维度不匹配的崩溃 Bug！
    // ////////////////////////////////////////////////////////////////////////
    bool has_conf_params = false;
    auto check_conf = [](const std::unordered_map<std::string, std::vector<double>>& m) {
        auto it = m.find("c_conf");
        return it != m.end() && !it->second.empty();
    };
    if (check_conf(user_params.free) || check_conf(user_params.fixed) || check_conf(user_params.constant)) {
        has_conf_params = true;
    }

    const auto& full_stim = df.at(col_stim);
    const auto& full_resp = df.at(col_resp);
    const std::vector<double>* full_conf_ptr = nullptr;
    const std::vector<double>* full_diff_ptr = nullptr;
    
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

    if (!col_diff.empty() && df.count(col_diff)) {
        full_diff_ptr = &df.at(col_diff);
    }

    std::vector<SubjectFitTask> tasks;

    // ////////////////////////////////////////////////////////////////////////
    // 3. 核心解耦：在这里把 R 语言里 foreach 的准备工作做完
    // 遍历每一个被试，提取属于 TA 的数据，生成独立的 freq 矩阵，
    // 并打包成 SubjectFitTask（完全独立的环境包）。
    // 这样多线程运行时，不同核心之间只读自己的内存，不需要加锁，极限加速！
    // ////////////////////////////////////////////////////////////////////////
    for (const auto& kv : info.subjects) {
        double subid = kv.first;
        const auto& subj = kv.second; 

        auto create_task = [&](
            const std::vector<int>& row_indices,
            const std::string& cond_name
        ) {
            // 根据被试所在行索引 (row_indices)，把该被试的具体数据抽出来
            std::vector<double> sub_stim, sub_resp, sub_conf, sub_diff;
            sub_stim.reserve(row_indices.size());
            sub_resp.reserve(row_indices.size());
            if (full_conf_ptr != nullptr) {
                sub_conf.reserve(row_indices.size());
            }
            if (full_diff_ptr != nullptr) {
                sub_diff.reserve(row_indices.size());
            }

            for (int idx : row_indices) {
                sub_stim.push_back(full_stim[idx]);
                sub_resp.push_back(full_resp[idx]);
                if (full_conf_ptr != nullptr) {
                    sub_conf.push_back((*full_conf_ptr)[idx]);
                }
                if (full_diff_ptr != nullptr) {
                    sub_diff.push_back((*full_diff_ptr)[idx]);
                }
            }

            size_t n_diffs = 1;
            if (full_diff_ptr != nullptr && !sub_diff.empty()) {
                std::set<double> diff_set(sub_diff.begin(), sub_diff.end());
                n_diffs = diff_set.size();
            }
            
            // ////////////////////////////////////////////////////////////////
            // 智能向量化拉伸：如果数据里有 3 个难度等级，就把 d 变成 c(d,d,d)
            // 优化器会为这3个 d 独立寻找最佳拟合值
            // ////////////////////////////////////////////////////////////////
            ParamGroup subj_user_params = user_params;
                if (!subj_user_params.free.count("d") && 
                    !subj_user_params.fixed.count("d") && 
                    !subj_user_params.constant.count("d")) {
                    subj_user_params.free["d"] = {1.5};
                }
                
                auto expand_param = [&](
                    std::unordered_map<std::string,
                    std::vector<double>>& m,
                    const std::string& key
                ) {
                    if (m.count(key) && !m[key].empty()) {
                        if (m[key].size() != n_diffs && n_diffs > 0) {
                            std::vector<double> new_vec(n_diffs, m[key][0]);
                            size_t bound = std::min(m[key].size(), n_diffs);
                            for (size_t k = 0; k < bound; ++k) {
                                new_vec[k] = m[key][k];
                            }
                            m[key] = new_vec;
                        }
                    }
                };
                expand_param(/*m=*/subj_user_params.free, /*key=*/"d");
                expand_param(/*m=*/subj_user_params.fixed, /*key=*/"d");
                expand_param(/*m=*/subj_user_params.constant, /*key=*/"d");
                
                ModifiedParamsResult subj_params = modify_params(
                    /*user_params=*/subj_user_params,
                    /*custom_lower=*/custom_lower,
                    /*custom_upper=*/custom_upper
                );

            // 预先算好频数矩阵 (freq_mat)，这是极大似然估计的基础。
            // 把耗时的数据分箱操作全都在外层做完，目标函数里只剩纯粹的数学乘法
            MatrixFreq freq_obj = matrix_freq(
                /*stim=*/sub_stim, 
                /*resp=*/sub_resp, 
                /*conf=*/full_conf_ptr != nullptr ? &sub_conf : nullptr, 
                /*diff=*/full_diff_ptr != nullptr ? &sub_diff : nullptr,
                /*std_params=*/&subj_params.flat
            );

            // 【封包】：这就是你的 R environment !
            SubjectFitTask task;
            task.subid = subid;
            task.cond = cond_name;
            task.model = model_name;
            task.freq = freq_obj;
            task.params = subj_params;
            tasks.push_back(task);
        };

        if (!subj.condition.empty()) {
            for (const auto& cond_kv : subj.condition) {
                create_task(cond_kv.second, cond_kv.first);
            }
        } else {
            create_task(subj.raw, "");
        }
    }

    return tasks;
}

// ////////////////////////////////////////////////////////////////////////////
// 目标函数 nll (Negative Log-Likelihood)
// 这是给 NLOPT 专属调用的回调函数。它的参数是定死的：
// @param n: 自由参数的数量 (相当于 R 里的 length(params))
// @param x: 当前这一步优化器给出的参数猜测值数组 (相当于 R 里的 params)
// @param grad: 梯度数组 (无需导数的算法可忽略，仅作占位)
// @param f_data: 魔术指针！这就是我们传进来的“局部环境”(SubjectFitTask)
// ////////////////////////////////////////////////////////////////////////////
double nll(unsigned n, const double* x, double* grad, void* f_data) {
    try {
        // ////////////////////////////////////////////////////////////////////
        // 1. 打开局部环境 (Unpack Environment)
        // 把虚无缥缈的 f_data 指针，还原成我们认识的 SubjectFitTask
        // 现在你可以从 task 里拿到 freq 矩阵和所有固定参数了！
        // ////////////////////////////////////////////////////////////////////
        SubjectFitTask* task = static_cast<SubjectFitTask*>(f_data);
        
        // ////////////////////////////////////////////////////////////////////
        // 2. 参数自动对齐 (相当于 R 里面的 mu = params[1])
        // 在 R 里你可能是硬编码：d = params[1]; c_resp = params[2] ...
        // 但在这里，系统会利用提前备好的字典结构 (task->params)，
        // 自动把 NLOPT 瞎猜的一维数组 x，按顺序塞进带有名字的字典 (std_params)。
        // 而那些固定参数 (fixed) 如 lapse=0.1 本来就在 flat 里，保持不动。
        // ////////////////////////////////////////////////////////////////////
        auto std_params = task->params.flat; // 局部拷贝，线程安全
        size_t x_idx = 0;
        std::vector<double> free_params_vec;
        for (const auto& name : task->params.name_free) {
            size_t param_len = task->params.structured.free.at(name).size();
            for (size_t i = 0; i < param_len; ++i) {
                double val = x[x_idx++];
                std_params[name][i] = val;
                free_params_vec.push_back(val);
            }
        }

        // ////////////////////////////////////////////////////////////////////
        // 3. 模型正向推导流水线
        // 环境齐全了，参数也有了名字，开始像工厂流水线一样计算！
        // ////////////////////////////////////////////////////////////////////
        std::vector<std::vector<double>> cdf_n;
        std::vector<std::vector<double>> cdf_s;

        if (task->model == "sdt") {
            ModelSDT<double> model(/*std_params=*/std_params);
            cdf_n = model.cdf_noise();
            cdf_s = model.cdf_signal();
        } else {
            throw std::invalid_argument(
                "Error: Unknown model name '" + task->model + "'."
            );
        }

        MatrixProb<double> prob = matrix_prob<double>(
            /*cdf_noise=*/cdf_n, /*cdf_signal=*/cdf_s, /*std_params=*/std_params
        );

        auto mult = matrix_mult<double>(
            /*freq_mat=*/task->freq.freq_mat,
            /*prob_mat=*/prob.prob_mat,
            /*std_params=*/std_params
        );

        auto loss = criterion_likelihood<double>(
            /*mult_mat=*/mult, 
            /*freq_mat=*/task->freq.freq_mat, 
            /*k=*/task->params.numb_free, 
            /*free_params=*/free_params_vec, 
            /*std_params=*/std_params
        );

        // ////////////////////////////////////////////////////////////////////
        // 4. 返回计算好的 NLL，交还给 NLOPT，指引它下一步瞎猜的方向
        // ////////////////////////////////////////////////////////////////////
        return loss.nll;
    } catch (const std::exception& e) {
        static thread_local bool error_printed = false;
        if (!error_printed) {
            #pragma omp critical
            {
                std::cerr << "\n[Fatal NLL Error] " << e.what() << "\n";
            }
            error_printed = true;
        }
        // C++ 防御底线：万一出错，返回一个极差的惩罚值，绝不能让 NLOPT 崩溃
        return 1e10;
    }
}