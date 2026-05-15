#include "../include/task_builder.hpp"
#include "../include/criterion_likelihood.hpp"
#include "../include/info_data.hpp"
#include "../include/matrix_mult.hpp"
#include "../include/matrix_prob.hpp"
#include "../include/model_sdt.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <set>
#include <stdexcept>

/* ========================================================================== *
 *                     Build Per-Subject Fitting Tasks                        *
 * ========================================================================== */

// 按被试(以及可选条件)构建拟合任务
// 该函数负责准备:
// 1) 被试级切片后的数据向量
// 2) 与数据形状对齐的参数容器
// 3) 用于似然计算的频数矩阵
// 4) 在启用 MAP 时所需的先验对象
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
    // 标准化并校验输入的列映射与分组信息
    DataInfoResult info = info_data(
        /*df=*/df,
        /*colnames=*/colnames
    );

    // 模型必须包含 stim 与 resp 两列, 否则无法计算
    if (!info.colnames.count("stim") || !info.colnames.count("resp")) {
        throw std::invalid_argument(
            "Error: Missing critical columns 'stim' or 'resp'."
        );
    }

    // 从标准化后的映射中取出规范列名
    std::string col_stim = info.colnames.at("stim");
    std::string col_resp = info.colnames.at("resp");

    // conf 与 diff 是可选列, 是否需要取决于参数配置
    std::string col_conf = "";
    if (info.colnames.count("conf")) {
        col_conf = info.colnames.at("conf");
    }

    std::string col_diff = "";
    if (info.colnames.count("diff")) {
        col_diff = info.colnames.at("diff");
    }

    // 检查任意参数块中是否声明了 c_conf
    // 若声明, 则数据中必须提供 confidence 列
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

    // 保存完整列的引用, 避免循环中重复 map 查找
    const auto& full_stim = df.at(col_stim);
    const auto& full_resp = df.at(col_resp);
    const std::vector<double>* full_conf_ptr = nullptr;
    const std::vector<double>* full_diff_ptr = nullptr;

    // 当模型包含 c_conf 参数时, 强制要求存在 conf 列
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

    // diff 列可缺省; 缺省时下游按单一难度层处理
    if (!col_diff.empty() && df.count(col_diff)) {
        full_diff_ptr = &df.at(col_diff);
    }

    // 汇总所有生成的"被试-条件"拟合任务
    std::vector<SubjectFitTask> tasks;

    // 遍历 info_data 产出的每个被试分组
    for (const auto& kv : info.subjects) {
        double subid = kv.first;
        const auto& subj = kv.second;

        // 该辅助函数为给定行索引集合创建一个任务
        // 单条件数据时 cond_name 为空字符串
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

            // 单次循环复制所需列, 保持各列行对齐
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

            // n_diffs 决定参数 d 的目标长度
            // 若无 diff 数据, 则退化为 1 个共享难度层
            size_t n_diffs = 1;
            if (full_diff_ptr != nullptr && !sub_diff.empty()) {
                std::set<double> diff_set(sub_diff.begin(), sub_diff.end());
                n_diffs = diff_set.size();
            }

            // 在局部副本上处理参数, 避免修改用户输入模板
            // 这样每个"被试-条件"都能独立调整参数形状
            ParamGroup subj_user_params = user_params;

            // 若 free/fixed/constant 中均未声明 d,
            // 则补一个默认自由参数, 保证 SDT 敏感度可估计
            if (!subj_user_params.free.count("d") &&
                !subj_user_params.fixed.count("d") &&
                !subj_user_params.constant.count("d")) {
                subj_user_params.free["d"] = {1.5};
            }

            // 将参数向量长度对齐到 n_diffs
            // 规则:
            // 1) 已对齐则不变
            // 2) 不足时用首元素扩展
            // 3) 若部分给定, 保留前面已给值
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

            // 检查自由参数边界, 避免优化器在无效区间上直接失败.
            const std::size_t n_free = static_cast<std::size_t>(
                std::max(subj_params.numb_free, 0)
            );
            for (std::size_t p_idx = 0; p_idx < n_free; ++p_idx) {
                if (subj_params.lower_bounds[p_idx] >=
                    subj_params.upper_bounds[p_idx]) {
                    std::cerr
                        << "\n[Warning] Boundary conflict detected for "
                        << "subject " << subid << " in condition '"
                        << cond_name << "'.\n"
                        << "Parameter index " << p_idx
                        << " has lower bound ("
                        << subj_params.lower_bounds[p_idx]
                        << ") >= upper bound ("
                        << subj_params.upper_bounds[p_idx] << ").\n";
                }
            }

            // 将行级数据转换为观测频数矩阵
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
