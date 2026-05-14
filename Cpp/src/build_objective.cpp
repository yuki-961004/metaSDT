#include "../include/build_objective.hpp"
#include "../include/criterion_likelihood.hpp"
#include "../include/info_data.hpp"
#include "../include/matrix_mult.hpp"
#include "../include/matrix_prob.hpp"
#include "../include/model_sdt.hpp"

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

/* ========================================================================== *
 *                     Build Per-Subject Fitting Tasks                        *
 * ========================================================================== */

// 按被试（以及可选条件）构建拟合任务。
// 该函数负责准备：
// 1) 被试级切片后的数据向量。
// 2) 与数据形状对齐的参数容器。
// 3) 用于似然计算的频数矩阵。
// 4) 在启用 MAP 时所需的先验对象。
std::vector<SubjectFitTask> build_fit_tasks(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper,
    bool apply_priors,
    const std::unordered_map<std::string, UserPrior>& user_priors
) {
    // 标准化并校验输入的列映射与分组信息。
    DataInfoResult info = info_data(
        /*df=*/df,
        /*colnames=*/colnames
    );

    // 模型必须包含 stim 与 resp 两列，否则无法计算。
    if (!info.colnames.count("stim") || !info.colnames.count("resp")) {
        throw std::invalid_argument(
            "Error: Missing critical columns 'stim' or 'resp'."
        );
    }

    // 从标准化后的映射中取出规范列名。
    std::string col_stim = info.colnames.at("stim");
    std::string col_resp = info.colnames.at("resp");

    // conf 与 diff 是可选列，是否需要取决于参数配置。
    std::string col_conf = "";
    if (info.colnames.count("conf")) {
        col_conf = info.colnames.at("conf");
    }

    std::string col_diff = "";
    if (info.colnames.count("diff")) {
        col_diff = info.colnames.at("diff");
    }

    // 检查任意参数块中是否声明了 c_conf。
    // 若声明，则数据中必须提供 confidence 列。
    bool has_conf_params = false;
    auto check_conf = [](
        const std::unordered_map<std::string, std::vector<double>>& m
    ) {
        auto it = m.find("c_conf");
        return it != m.end() && !it->second.empty();
    };
    if (check_conf(user_params.free) ||
        check_conf(user_params.fixed) ||
        check_conf(user_params.constant)) {
        has_conf_params = true;
    }

    // 保存完整列的引用，避免循环中重复 map 查找。
    const auto& full_stim = df.at(col_stim);
    const auto& full_resp = df.at(col_resp);
    const std::vector<double>* full_conf_ptr = nullptr;
    const std::vector<double>* full_diff_ptr = nullptr;

    // 当模型包含 c_conf 参数时，强制要求存在 conf 列。
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

    // diff 列可缺省；缺省时下游按单一难度层处理。
    if (!col_diff.empty() && df.count(col_diff)) {
        full_diff_ptr = &df.at(col_diff);
    }

    // 汇总所有生成的“被试-条件”拟合任务。
    std::vector<SubjectFitTask> tasks;

    // 遍历 info_data 产出的每个被试分组。
    for (const auto& kv : info.subjects) {
        double subid = kv.first;
        const auto& subj = kv.second;

        // 该辅助函数为给定行索引集合创建一个任务。
        // 单条件数据时 cond_name 为空字符串。
        auto create_task = [&](const std::vector<int>& row_indices,
                               const std::string& cond_name) {
            // 将原始列按当前“被试-条件”行索引切片。
            std::vector<double> sub_stim;
            std::vector<double> sub_resp;
            std::vector<double> sub_conf;
            std::vector<double> sub_diff;

            // 预分配容量，减少循环内重复扩容开销。
            sub_stim.reserve(row_indices.size());
            sub_resp.reserve(row_indices.size());
            if (full_conf_ptr != nullptr) {
                sub_conf.reserve(row_indices.size());
            }
            if (full_diff_ptr != nullptr) {
                sub_diff.reserve(row_indices.size());
            }

            // 单次循环复制所需列，保持各列行对齐。
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

            // n_diffs 决定参数 d 的目标长度。
            // 若无 diff 数据，则退化为 1 个共享难度层。
            size_t n_diffs = 1;
            if (full_diff_ptr != nullptr && !sub_diff.empty()) {
                std::set<double> diff_set(sub_diff.begin(), sub_diff.end());
                n_diffs = diff_set.size();
            }

            // 在局部副本上处理参数，避免修改用户输入模板。
            // 这样每个“被试-条件”都能独立调整参数形状。
            ParamGroup subj_user_params = user_params;

            // 若 free/fixed/constant 中均未声明 d，
            // 则补一个默认自由参数，保证 SDT 敏感度可估计。
            if (!subj_user_params.free.count("d") &&
                !subj_user_params.fixed.count("d") &&
                !subj_user_params.constant.count("d")) {
                subj_user_params.free["d"] = {1.5};
            }

            // 将参数向量长度对齐到 n_diffs。
            // 规则：
            // 1) 已对齐则不变。
            // 2) 不足时用首元素扩展。
            // 3) 若部分给定，保留前面已给值。
            auto expand_param = [&n_diffs](
                std::unordered_map<std::string, std::vector<double>>& m,
                const std::string& key
            ) {
                if (!m.count(key) || m[key].empty()) {
                    return;
                }
                if (m[key].size() == n_diffs || n_diffs == 0) {
                    return;
                }
                std::vector<double> new_vec(n_diffs, m[key][0]);
                size_t bound = std::min(m[key].size(), n_diffs);
                for (size_t k = 0; k < bound; ++k) {
                    new_vec[k] = m[key][k];
                }
                m[key] = new_vec;
            };

            // 对 free/fixed/constant 三个参数块执行同一对齐逻辑。
            expand_param(/*m=*/subj_user_params.free, /*key=*/"d");
            expand_param(/*m=*/subj_user_params.fixed, /*key=*/"d");
            expand_param(/*m=*/subj_user_params.constant, /*key=*/"d");

            // 生成扁平参数结构以及优化器上下界。
            ModifiedParamsResult subj_params = modify_params(
                /*user_params=*/subj_user_params,
                /*custom_lower=*/custom_lower,
                /*custom_upper=*/custom_upper
            );

            // 将行级数据转换为观测频数矩阵。
            MatrixFreq freq_obj = matrix_freq(
                /*stim=*/sub_stim,
                /*resp=*/sub_resp,
                /*conf=*/full_conf_ptr != nullptr ? &sub_conf : nullptr,
                /*diff=*/full_diff_ptr != nullptr ? &sub_diff : nullptr,
                /*std_params=*/&subj_params.flat
            );

            // 组装后续优化阶段直接使用的任务对象。
            SubjectFitTask task;
            task.subid = subid;
            task.cond = cond_name;
            task.model = model_name;
            task.freq = freq_obj;
            task.params = subj_params;
            task.prior = modify_prior(user_priors, subj_params, apply_priors);
            tasks.push_back(task);
        };

        // 若存在条件分组，则每个条件创建一个任务。
        // 否则该被试只创建一个总任务。
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

/* ========================================================================== *
 *                      Negative Log-Likelihood Objective                     *
 * ========================================================================== */

// NLopt 的目标函数回调。
// 输入 x 仅包含按顺序展开的自由参数。
// 函数会重建完整参数、计算似然与先验，
// 最终返回负对数后验：nll - log_prior。
double nll(unsigned n, const double* x, double* grad, void* f_data) {
    try {
        // 取回通过 NLopt 回调数据传入的任务指针。
        SubjectFitTask* task = static_cast<SubjectFitTask*>(f_data);

        // 从模板参数图出发，用 x 覆盖自由参数位置。
        auto std_params = task->params.flat;
        size_t x_idx = 0;
        std::vector<double> free_params_vec;

        // 按扁平向量 x 回填结构化自由参数。
        // 循环顺序与 name_free 一致，以匹配边界顺序。
        for (const auto& name : task->params.name_free) {
            size_t param_len = task->params.structured.free.at(name).size();
            for (size_t i = 0; i < param_len; ++i) {
                double val = x[x_idx++];
                std_params[name][i] = val;
                free_params_vec.push_back(val);
            }
        }

        // 对 c_conf 排序，保证置信阈值单调有序。
        auto it_c_conf = std_params.find("c_conf");
        if (it_c_conf != std_params.end() && !it_c_conf->second.empty()) {
            std::sort(it_c_conf->second.begin(), it_c_conf->second.end());
        }

        std::vector<std::vector<double>> cdf_n;
        std::vector<std::vector<double>> cdf_s;

        // 按模型名称分派对应的 CDF 生成逻辑。
        if (task->model == "sdt") {
            ModelSDT<double> model(/*std_params=*/std_params);
            cdf_n = model.cdf_noise();
            cdf_s = model.cdf_signal();
        } else {
            throw std::invalid_argument(
                "Error: Unknown model name '" + task->model + "'."
            );
        }

        // 将 CDF 转换为各响应类别的概率矩阵。
        MatrixProb<double> prob = matrix_prob<double>(
            /*cdf_noise=*/cdf_n,
            /*cdf_signal=*/cdf_s,
            /*std_params=*/std_params
        );

        // 结合观测频数与理论概率，构造似然计算输入。
        auto mult = matrix_mult<double>(
            /*freq_mat=*/task->freq.freq_mat,
            /*prob_mat=*/prob.prob_mat,
            /*std_params=*/std_params
        );

        // 计算似然准则，得到 NLL 等统计量。
        auto loss = criterion_likelihood<double>(
            /*mult_mat=*/mult,
            /*freq_mat=*/task->freq.freq_mat,
            /*k=*/task->params.numb_free,
            /*free_params=*/free_params_vec,
            /*std_params=*/std_params
        );

        // 先验评估器需要 Eigen::VectorXd，因此先做类型转换。
        Eigen::VectorXd eigen_free_params(free_params_vec.size());
        for (size_t i = 0; i < free_params_vec.size(); ++i) {
            eigen_free_params(static_cast<Eigen::Index>(i)) =
                free_params_vec[i];
        }

        // MAP 目标函数：负对数似然减去对数先验。
        double log_prior = task->prior.evaluate<double>(eigen_free_params);
        return loss.nll - log_prior;
    } catch (const std::exception& e) {
        // 每个线程只打印一次错误，避免并行时日志刷屏。
        static thread_local bool error_printed = false;
        if (!error_printed) {
#pragma omp critical
            {
                std::cerr << "\n[Fatal NLL Error] " << e.what() << "\n";
            }
            error_printed = true;
        }

        // 显式忽略未使用参数，避免编译器告警。
        (void)n;
        (void)grad;

        // 返回大惩罚值，让优化器远离出错参数区域。
        return 1e10;
    }
}
