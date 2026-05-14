#include "../include/estimate_map.hpp"
#include "../include/algorithm_nlopt.hpp"
#include "../include/build_objective.hpp"
#include "../include/criterion_likelihood.hpp"
#include "../include/criterion_posterior.hpp"
#include "../include/info_nlopt.hpp"
#include "../include/matrix_mult.hpp"
#include "../include/matrix_prob.hpp"
#include "../include/model_sdt.hpp"
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

namespace {

/* ========================================================================== *
 *                              Internal Helpers                              *
 * ========================================================================== */

// 保存单个条件运行结果的结构体
struct CondRunResult {
    std::vector<SubjectFitResult> results;
    int iter_used = 0;
    std::string stop_reason = "not_started";
};

/* ========================================================================== *
 *                         Expectation Step (E-Step)                          *
 * ========================================================================== */

// 通过独立拟合每个任务的模型来执行 E 步
std::vector<SubjectFitResult> em_e_step(
    std::vector<SubjectFitTask>& tasks,
    const NLoptControl& control,
    const std::string& progress_title = ""
) {
    std::vector<SubjectFitResult> results(tasks.size());
    const int n_tasks = static_cast<int>(tasks.size());

    // 如果启用了打印且存在任务，则初始化进度条
    if (control.print_level > 0 && n_tasks > 0 && !progress_title.empty()) {
        ui::ProgressOptions popts;
        popts.mode = control.progress;
        popts.refresh_ms = control.progress_refresh_ms;
        popts.line_interval_sec = control.progress_line_interval_sec;
        popts.line_interval_pct = control.progress_line_interval_pct;
        ui::progress_start(
            static_cast<std::size_t>(n_tasks),
            progress_title,
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

            const std::vector<double> free_params_vec =
                task.params.extract_free_vector(best_p);
            Eigen::VectorXd free_params(
                static_cast<Eigen::Index>(free_params_vec.size())
            );
            
            // 将标准向量转换为 Eigen 向量以进行数学运算
            for (size_t p_idx = 0; p_idx < free_params_vec.size(); ++p_idx) {
                free_params(static_cast<Eigen::Index>(p_idx)) =
                    free_params_vec[p_idx];
            }

            const CriterionPosterior posterior(
                task.freq.freq_mat,
                task.params.name_free,
                task.params.get_free_sizes(),
                task.params.flat,
                task.prior
            );
            const double log_post = posterior.operator()<double>(free_params);

            std::vector<std::vector<double>> cdf_n;
            std::vector<std::vector<double>> cdf_s;
            
            // 根据模型生成累积分布函数
            if (task.model == "sdt") {
                ModelSDT<double> model_obj(best_p);
                cdf_n = model_obj.cdf_noise();
                cdf_s = model_obj.cdf_signal();
            } else {
                throw std::invalid_argument(
                    "Error: Unknown model name '" + task.model + "'."
                );
            }

            MatrixProb<double> prob = matrix_prob<double>(cdf_n, cdf_s, best_p);
            const auto mult = matrix_mult<double>(
                task.freq.freq_mat,
                prob.prob_mat,
                best_p
            );
            const auto loss = criterion_likelihood<double>(
                mult,
                task.freq.freq_mat,
                task.params.numb_free,
                free_params_vec,
                best_p
            );

            res.logL = loss.logL;
            res.logPost = log_post;
            res.logPrior = res.logPost - res.logL;

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
            
            res.best_params = best_p;
            
        } catch (const std::exception& e) {
            // 针对失败优化的线程安全错误打印
            #pragma omp critical
            {
                std::cerr << "\n[NLOPT Error] Subject " << task.subid
                          << " MAP fitting failed: " << e.what() << "\n";
            }
            res.status = -1;
            res.n_evals = 0;
            res.result_message = e.what();
            res.stop_reason = "exception";
        }

        results[i] = res;
        
        // 安全地更新进度条
        if (control.print_level > 0 && !progress_title.empty()) {
            ui::progress_advance(1);
        }
    }

    // 完成所有任务后清理进度条
    if (control.print_level > 0 && n_tasks > 0 && !progress_title.empty()) {
        ui::progress_finish();
    }

    return results;
}

/* ========================================================================== *
 *                         Maximization Step (M-Step)                         *
 * ========================================================================== */

// 管理分层期望最大化迭代
CondRunResult em_m_step(
    std::vector<SubjectFitTask> tasks,
    const NLoptControl& control,
    const std::string& condition_col_name
) {
    CondRunResult out;
    
    // 如果没有提供任务，则提前返回
    if (tasks.empty()) {
        out.stop_reason = "empty";
        return out;
    }

    // 如果需要，使用最大似然估计进行初始化
    if (control.em_init_mle) {
        std::vector<SubjectFitTask> mle_tasks = tasks;
        
        // 移除 MLE 初始化的先验分布
        for (auto& t : mle_tasks) {
            t.prior = CriterionPrior();
        }
        
        const std::vector<SubjectFitResult> mle_res = em_e_step(
            mle_tasks,
            control
        );
        
        // 使用 MLE 结果更新任务的最佳参数
        for (size_t i = 0; i < tasks.size() && i < mle_res.size(); ++i) {
            optimal(tasks[i], mle_res[i]);
        }
    }

    std::vector<SubjectFitResult> best_results = em_e_step(tasks, control);
    double best_sum_logpost = 0.0;
    
    // 计算初始拟合的总对数后验概率
    for (const auto& r : best_results) {
        best_sum_logpost += r.logPost;
    }

    double prev_sum_logpost = best_sum_logpost;
    const bool use_patience = control.em_patience > 0;
    int patience_left = control.em_patience;

    out.results = best_results;
    out.stop_reason = "init_only";

    // 如果需要，初始化 EM 循环的可视化进度跟踪
    if (control.print_level > 0 && control.em_max_iter > 0) {
        std::string title = "MAP";
        
        // 将条件详细信息附加到进度条标题
        if (!condition_col_name.empty()) {
            title += " " + condition_col_name;
        }
        if (!tasks.empty() && !tasks[0].cond.empty()) {
            title += " [" + tasks[0].cond + "]";
        }

        ui::ProgressOptions popts;
        popts.mode = control.progress;
        popts.refresh_ms = control.progress_refresh_ms;
        popts.line_interval_sec = control.progress_line_interval_sec;
        popts.line_interval_pct = control.progress_line_interval_pct;
        ui::progress_start(
            static_cast<std::size_t>(control.em_max_iter),
            title,
            control.progress_refresh_ms,
            popts
        );
    }

    // 主要 EM 迭代循环
    for (int iter = 0; iter < control.em_max_iter; ++iter) {
        out.iter_used = iter + 1;
        
        // 更新 EM 迭代进度条
        if (control.print_level > 0) {
            ui::progress_set(static_cast<std::size_t>(out.iter_used));
        }

        std::unordered_map<size_t, std::vector<std::vector<double>>> grouped;
        
        // 基于自由参数的数量跨被试对参数进行分组
        for (size_t i = 0; i < tasks.size(); ++i) {
            std::vector<double> fp = tasks[i].params.extract_free_vector(
                best_results[i].best_params
            );
            grouped[fp.size()].push_back(fp);
            optimal(tasks[i], best_results[i]);
        }

        // 使用分组的被试参数分布更新经验先验
        for (auto& task : tasks) {
            const size_t k = static_cast<size_t>(task.params.numb_free);
            const auto it = grouped.find(k);
            
            // 仅在存在足够的总体方差时更新先验
            if (it != grouped.end() && it->second.size() >= 2) {
                task.prior.update(it->second);
            }
        }

        std::vector<SubjectFitResult> current = em_e_step(tasks, control);

        double sum_logpost = 0.0;
        // 计算新的聚合对数后验似然
        for (const auto& r : current) {
            sum_logpost += r.logPost;
        }
        
        const double delta = sum_logpost - prev_sum_logpost;

        // 基于阈值容差检查收敛性
        if (std::abs(delta) <= control.em_tol) {
            out.results = current;
            out.stop_reason = "em_tol";
            
            if (control.print_level > 0 && control.em_max_iter > 0) {
                ui::progress_finish();
            }
            return out;
        }

        // 如果目标值增加则保留最佳结果，否则减少耐心值
        if (sum_logpost > best_sum_logpost) {
            best_sum_logpost = sum_logpost;
            best_results = current;
            out.results = current;
            
            // 改进时动态恢复耐心值
            if (use_patience) {
                patience_left = std::min(
                    control.em_patience,
                    patience_left + 1
                );
            }
        } else {
            // 当模型没有改进时消耗耐心阈值
            if (use_patience) {
                patience_left -= 1;
            }
        }

        prev_sum_logpost = sum_logpost;

        // 如果连续迭代没有产生改进，则终止
        if (use_patience && patience_left <= 0) {
            out.stop_reason = "em_patience";
            if (control.print_level > 0 && control.em_max_iter > 0) {
                ui::progress_finish();
            }
            return out;
        }
    }

    out.stop_reason = "em_max_iter";
    if (control.print_level > 0 && control.em_max_iter > 0) {
        ui::progress_finish();
    }
    return out;
}

} // namespace

/* ========================================================================== *
 *                              Main Public API                               *
 * ========================================================================== */

// 执行最大后验 (MAP) 估计的入口点
std::vector<SubjectFitResult> estimate_map(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const NLoptControl& raw_control,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper,
    const std::unordered_map<std::string, UserPrior>& user_priors,
    int* em_iterations_used,
    std::unordered_map<std::string, int>* em_iterations_by_cond,
    std::unordered_map<std::string, std::string>* em_stop_reason_by_cond
) {
    const NLoptControl control = modify_control(raw_control, "map");

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
        df,
        colnames,
        user_params,
        model_name,
        custom_lower,
        custom_upper,
        true,
        user_priors
    );

    // 如果没有找到有效目标，则跳过计算并返回空结果
    if (tasks.empty()) {
        return {};
    }

    std::unordered_map<std::string, std::vector<SubjectFitTask>> tasks_by_cond;
    // 在给定的实验条件下适当地映射任务
    for (const auto& t : tasks) {
        tasks_by_cond[t.cond].push_back(t);
    }

    std::string condition_col_name;
    const auto it_cond = colnames.find("condition");
    // 如果命名映射中可用，则分配适当的列条件
    if (it_cond != colnames.end()) {
        condition_col_name = it_cond->second;
    }

    std::vector<SubjectFitResult> all_results;
    int total_iter = 0;

    // 顺序处理按条件层聚类的任务 
    for (auto& kv : tasks_by_cond) {
        CondRunResult cond_res = em_m_step(
            kv.second,
            control,
            condition_col_name
        );
        total_iter += cond_res.iter_used;
        
        all_results.insert(
            all_results.end(),
            cond_res.results.begin(),
            cond_res.results.end()
        );
        
        // 如果绑定了追踪器，则记录各个条件的迭代次数
        if (em_iterations_by_cond != nullptr) {
            (*em_iterations_by_cond)[kv.first] = cond_res.iter_used;
        }
        
        // 如果绑定了追踪器，则输出每个条件的详细退出状态
        if (em_stop_reason_by_cond != nullptr) {
            (*em_stop_reason_by_cond)[kv.first] = cond_res.stop_reason;
        }
    }

    // 累加所有层的总迭代次数以提供全局摘要
    if (em_iterations_used != nullptr) {
        *em_iterations_used = total_iter;
    }

    return all_results;
}
