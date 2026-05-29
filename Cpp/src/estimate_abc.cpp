#include "../include/estimate_abc.hpp"
#include "../include/algorithm_abcpp.hpp"
#include "../include/modify_control.hpp"
#include "../include/task_builder.hpp"
#include "../include/progress_bar.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ========================================================================== *
 *                              Main Public API                               *
 * ========================================================================== */

// 执行近似贝叶斯计算 (ABC) 估计的入口点
std::vector<SubjectABCResult> estimate_abc(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const ABCControl& raw_control,
    const std::unordered_map<std::string, UserPrior>& user_priors
) {
    // 规范化控制参数配置并获取基础参数
    const ABCControl control = modify_control(raw_control, "abc");
    const auto prior_map = abcppAdapter::merged_priors(user_priors);

    // 解析数据并构建受试者层级的拟合任务
    std::vector<SubjectFitTask> tasks = build_fit_tasks(
        df,
        colnames,
        user_params,
        model_name
    );

    if (tasks.empty()) {
        return {};
    }

    const int n_tasks = static_cast<int>(tasks.size());
    std::vector<SubjectABCResult> results(static_cast<std::size_t>(n_tasks));

#ifdef _OPENMP
    // ABC 还没有类似 MLE 的动态线程控制属性, 但可以预留扩展
    // 默认使用 OpenMP 分配的全局线程资源
#endif

    // 如果启用了打印且存在任务, 则初始化进度条
    if (control.print_level > 0 && n_tasks > 0) {
        ui::ProgressOptions popts;
        popts.mode = "dynamic"; 
        ui::progress_start(
            static_cast<std::size_t>(n_tasks),
            "ABC",
            100,
            popts
        );
    }

    // 顺序处理每个拟合任务
    #pragma omp parallel for
    for (int i = 0; i < n_tasks; ++i) {
        const auto& task = tasks[static_cast<std::size_t>(i)];
        SubjectABCResult out;

        try {
            out = abcppAdapter::run_subject_abc(
                task,
                control,
                prior_map,
                i
            );
        } catch (const std::exception& e) {
            // 捕捉并记录任何由模型计算或拟合过程中抛出的错误
            out.subid = task.subid;
            out.cond = task.cond;
            out.status = -1;
            out.message = e.what();
            
            // 线程安全地打印错误信息
            #pragma omp critical
            {
                std::cerr << "\n[ABC Error] Subject " << task.subid
                          << " fitting failed: " << e.what() << "\n";
            }
        }

        results[static_cast<std::size_t>(i)] = out;

        if (control.print_level > 0) {
            #pragma omp critical
            {
                ui::progress_advance(1);
            }
        }
    }

    if (control.print_level > 0 && n_tasks > 0) {
        ui::progress_finish();
    }

    return results;
}
