#include "../include/estimate_mle.hpp"
#include "../include/algorithm_nlopt.hpp"
#include "../include/info_nlopt.hpp"
#include "../include/modify_control.hpp"
#include "../include/progress_bar.hpp"

#include <nlopt.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ========================================================================== *
 *                              Main Public API                               *
 * ========================================================================== */

// 执行最大似然 (MLE) 估计的入口点
std::vector<SubjectFitResult> estimate_mle(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const NLoptControl& raw_control,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper
) {
    const NLoptControl control = modify_control(raw_control, "mle");

#ifdef _OPENMP
    // 如果有指定，动态定义全局核心利用率
    if (control.threads > 0) {
        omp_set_num_threads(control.threads);
    }
#endif

    // 全局设置 RNG 种子以实现可重复的优化
    if (control.seed >= 0) {
        nlopt::srand(static_cast<unsigned long>(control.seed));
    }

    std::vector<SubjectFitTask> tasks = build_fit_tasks(
        /*df=*/df,
        /*colnames=*/colnames,
        /*user_params=*/user_params,
        /*model_name=*/model_name,
        /*custom_lower=*/custom_lower,
        /*custom_upper=*/custom_upper
    );

    std::vector<SubjectFitResult> results(tasks.size());
    const int n_tasks = static_cast<int>(tasks.size());

    // 如果启用了打印且存在任务，则初始化进度条
    if (control.print_level > 0 && n_tasks > 0) {
        ui::ProgressOptions popts;
        popts.mode = control.progress;
        popts.refresh_ms = control.progress_refresh_ms;
        popts.line_interval_sec = control.progress_line_interval_sec;
        popts.line_interval_pct = control.progress_line_interval_pct;
        ui::progress_start(
            static_cast<std::size_t>(n_tasks),
            "MLE",
            control.progress_refresh_ms,
            popts
        );
    }

    // 如果 OpenMP 可用，则跨被试并行化任务执行
    #pragma omp parallel for
    for (int i = 0; i < n_tasks; ++i) {
        auto& task = tasks[i];
        SubjectFitResult res;
        res.subid = task.subid;
        res.cond = task.cond;
        res.best_params = task.params.flat;

        // 将所有自由参数的初始值收集到一个向量中
        std::vector<double> x0 = task.params.extract_free_vector(
            task.params.flat
        );

        // 确保初始点位于边界内
        sanitize_initial_point(
            x0,
            task.params.lower_bounds,
            task.params.upper_bounds
        );

        try {
            // 使用控制设置和边界配置 NLopt 优化器
            nlopt::opt opt = create_nlopt_optimizer(
                control,
                task.params.lower_bounds,
                task.params.upper_bounds
            );
            opt.set_min_objective(nll, &task);

            double minf = 0.0;
            nlopt::result nlopt_res = nlopt::FAILURE;
            
            try {
                nlopt_res = opt.optimize(x0, minf);
                res.status = static_cast<int>(nlopt_res);
                res.n_evals = opt.get_numevals();
            } catch (const std::exception& e) {
                // 在本地处理此任务的优化异常
                res.status = -1;
                res.n_evals = opt.get_numevals();
                res.result_message = e.what();
                res.stop_reason = "exception";
                throw;
            }

            res.optimum_value = minf;
            const NLoptStatusInfo nlopt_info = nlopt_status_info(nlopt_res);
            res.result_message = nlopt_info.message;
            res.stop_reason = nlopt_info.stop_reason;

            // 检查优化器是否未能启动
            if (static_cast<int>(nlopt_res) < 0) {
                #pragma omp critical
                {
                    std::cerr
                        << "\n[NLOPT Error] Subject " << task.subid
                        << " failed to start optimization. NLopt Code: "
                        << static_cast<int>(nlopt_res)
                        << " (Check if maxeval>0 and initial parameters "
                        << "are strictly inside bounds).\n";
                }
            }

            res.logL = -minf;

            double N = 0.0;
            // 累加总频率计数以用于 AIC/BIC 计算
            for (const auto& dim_mat : task.freq.freq_mat) {
                // 遍历二维矩阵中的每一行
                for (const auto& row : dim_mat) {
                    // 遍历行中的每个单元格
                    for (const double v : row) {
                        N += v;
                    }
                }
            }

            res.aic = 2.0 * task.params.numb_free - 2.0 * res.logL;
            
            // 如果 N 为非正数，则处理 BIC 的边缘情况
            if (N > 0.0) {
                res.bic = task.params.numb_free * std::log(N) - 2.0 * res.logL;
            } else {
                res.bic = 0.0;
            }

            auto best_p = task.params.flat;
            
            // 将优化后的扁平向量映射回参数结构
            task.params.update_map_from_free_vector(best_p, x0);

            // 确保置信度标准严格排序
            if (best_p.count("c_conf")) {
                std::sort(best_p["c_conf"].begin(), best_p["c_conf"].end());
            }
            
            // 如果需要排序，则对标准差 'd' 进行排序
            if (best_p.count("d") && best_p.count("sort_d") &&
                !best_p["sort_d"].empty() && best_p["sort_d"][0] != 0.0) {
                std::sort(best_p["d"].rbegin(), best_p["d"].rend());
            }

            res.best_params = best_p;
            
        } catch (const std::exception& e) {
            // 针对失败优化的线程安全错误打印
            #pragma omp critical
            {
                std::cerr << "\n[NLOPT Error] Subject " << task.subid
                          << " fitting failed: " << e.what() << "\n";
            }
            
            // 如果状态没有被设置为错误，则设置为 -1
            if (res.status == 0) {
                res.status = -1;
            }
            
            // 如果没有结果信息，则使用异常信息
            if (res.result_message.empty()) {
                res.result_message = e.what();
            }
            
            // 如果停止原因没有设置，则标记为异常
            if (res.stop_reason.empty()) {
                res.stop_reason = "exception";
            }
        }

        results[i] = res;
        
        // 安全地更新进度条
        if (control.print_level > 0) {
            ui::progress_advance(1);
        }
    }

    // 完成所有任务后清理进度条
    if (control.print_level > 0 && n_tasks > 0) {
        ui::progress_finish();
    }

    return results;
}
