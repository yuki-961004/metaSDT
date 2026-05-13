#include "../include/estimate_mle.hpp"
#include "../include/modify_control.hpp"
#include "../include/progress_bar.hpp"
#include <nlopt.h>               // 直接使用系统内置的纯 C 语言头文件，彻底抛弃脆弱的 C++ 包装器
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sstream>

// 引入 OpenMP 支持多线程并行
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
std::string nlopt_result_message(nlopt_result code) {
    switch (code) {
        case NLOPT_SUCCESS: return "Success";
        case NLOPT_STOPVAL_REACHED: return "Stop value reached";
        case NLOPT_FTOL_REACHED: return "Function tolerance reached";
        case NLOPT_XTOL_REACHED: return "Parameter tolerance reached";
        case NLOPT_MAXEVAL_REACHED: return "Maximum evaluations reached";
        case NLOPT_MAXTIME_REACHED: return "Maximum time reached";
        case NLOPT_FAILURE: return "Generic failure";
        case NLOPT_INVALID_ARGS: return "Invalid arguments";
        case NLOPT_OUT_OF_MEMORY: return "Out of memory";
        case NLOPT_ROUNDOFF_LIMITED: return "Roundoff limited progress";
        case NLOPT_FORCED_STOP: return "Forced stop";
        default: return "Unknown result";
    }
}

std::string nlopt_stop_reason(nlopt_result code) {
    switch (code) {
        case NLOPT_STOPVAL_REACHED: return "stopval";
        case NLOPT_FTOL_REACHED: return "ftol";
        case NLOPT_XTOL_REACHED: return "xtol";
        case NLOPT_MAXEVAL_REACHED: return "maxeval";
        case NLOPT_MAXTIME_REACHED: return "maxtime";
        case NLOPT_SUCCESS: return "success";
        case NLOPT_ROUNDOFF_LIMITED: return "roundoff_limited";
        case NLOPT_FORCED_STOP: return "forced_stop";
        default: return "failure";
    }
}
}

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
    if (control.threads > 0) {
        omp_set_num_threads(control.threads);
    }
#endif
    if (control.seed >= 0) {
        nlopt_srand(static_cast<unsigned long>(control.seed));
    }
    // 1. 任务工厂：瞬间生成所有被试的独立环境包
    std::vector<SubjectFitTask> tasks = build_fit_tasks(
        /*df=*/df, 
        /*colnames=*/colnames, 
        /*user_params=*/user_params, 
        /*model_name=*/model_name, 
        /*custom_lower=*/custom_lower,
        /*custom_upper=*/custom_upper
    );
    
    // 预分配结果向量，避免多线程 push_back 导致的内存冲突
    std::vector<SubjectFitResult> results(
        tasks.size()
    );

    // 2. 多核并行启动！每个线程接管一个被试
    // (如果编译器支持并开启了 OpenMP，这里将把几百个被试自动分配给所有 CPU 核心)
    int n_tasks = static_cast<int>(tasks.size());
    if (control.print_level > 0 && n_tasks > 0) {
        ui::ProgressOptions popts;
        popts.mode = control.progress;
        popts.refresh_ms = control.progress_refresh_ms;
        popts.line_interval_sec = control.progress_line_interval_sec;
        popts.line_interval_pct = control.progress_line_interval_pct;
        ui::progress_start(static_cast<std::size_t>(n_tasks), "MLE", control.progress_refresh_ms, popts);
    }
    #pragma omp parallel for
    for (int i = 0; i < n_tasks; ++i) {
        auto& task = tasks[i];
        SubjectFitResult res;
        res.subid = task.subid;
        res.cond = task.cond;

        // 预设 base_params，防止发生异常时 best_params 为空导致 R 端展平报错
        res.best_params = task.params.flat;

        // ==========================================================
        // A. 提取初始参数猜测值 (Initial Guess x0)
        // ==========================================================
        std::vector<double> x0;
        x0.reserve(task.params.numb_free);
        for (const auto& name : task.params.name_free) {
            const auto& vals = task.params.structured.free.at(name);
            for (double v : vals) {
                x0.push_back(v);
            }
        }

        // 防御性编程：强制把初始值修剪在边界内，防止用户给的初始值超界直接报错
        for (size_t j = 0; j < x0.size(); ++j) {
            if (x0[j] <= task.params.lower_bounds[j]) {
                x0[j] = task.params.lower_bounds[j] + 1e-4;
            }
            if (x0[j] >= task.params.upper_bounds[j]) {
                x0[j] = task.params.upper_bounds[j] - 1e-4;
            }
        }

        // ==========================================================
        // B. 实例化该线程专属的 NLOPT 优化器并运行
        // ==========================================================
        try {
            // 动态选择算法：利用 NLopt 官方提供的字符串解析函数
            nlopt_algorithm algo = nlopt_algorithm_from_string(
                control.algorithm.c_str()
            );
            if (static_cast<int>(algo) < 0) {
                throw std::invalid_argument(
                    "Error: Invalid NLopt algorithm name: '" + control.algorithm + "'"
                );
            }

            nlopt_opt opt = nlopt_create(
                /*algorithm=*/algo, /*n=*/task.params.numb_free
            );
            
            // 完美对接你在 modify_params 铺好的边界
            nlopt_set_lower_bounds(/*opt=*/opt, /*lb=*/task.params.lower_bounds.data());
            nlopt_set_upper_bounds(/*opt=*/opt, /*ub=*/task.params.upper_bounds.data());

            // 如果用户选择了 MLSL 等依赖局部优化器的全局算法，动态创建并挂载附属优化器
            nlopt_opt local_opt = NULL;
            if (control.algorithm.find("MLSL") != std::string::npos || 
                control.algorithm.find("AUGLAG") != std::string::npos) {
                
                nlopt_algorithm local_algo = nlopt_algorithm_from_string(
                    control.local_algorithm.c_str()
                );
                if (static_cast<int>(local_algo) < 0) {
                    throw std::invalid_argument(
                        "Error: Invalid local NLopt algorithm name: '" + 
                        control.local_algorithm + "'"
                    );
                }
                
                local_opt = nlopt_create(
                    /*algorithm=*/local_algo, /*n=*/task.params.numb_free
                );
                
                // 将局部优化器的容差标准与主优化器保持一致
                nlopt_set_xtol_rel(/*opt=*/local_opt, /*tol=*/control.xtol_rel);
                if (control.ftol_rel > 0) {
                    nlopt_set_ftol_rel(/*opt=*/local_opt, /*tol=*/control.ftol_rel);
                }
                if (control.ftol_abs > 0) {
                    nlopt_set_ftol_abs(/*opt=*/local_opt, /*tol=*/control.ftol_abs);
                }
                if (control.xtol_abs > 0) {
                    nlopt_set_xtol_abs1(/*opt=*/local_opt, /*tol=*/control.xtol_abs);
                }
                
                // 挂载到全局优化器上
                nlopt_set_local_optimizer(/*opt=*/opt, /*local_opt=*/local_opt);
            }
            
            // 将 nll 静态函数与当前被试的内存地址(&task) 绑定
            nlopt_set_min_objective(/*opt=*/opt, /*f=*/nll, /*f_data=*/&task);
            
            // 设定优化停止的容差和最大迭代次数
            nlopt_set_xtol_rel(/*opt=*/opt, /*tol=*/control.xtol_rel);
            nlopt_set_maxeval(/*opt=*/opt, /*maxeval=*/control.maxeval); 
            
            // 开放更多高级控制接口 (当且仅当用户指定了>0的值时启用)
            if (control.ftol_rel > 0) {
                nlopt_set_ftol_rel(/*opt=*/opt, /*tol=*/control.ftol_rel);
            }
            if (control.ftol_abs > 0) {
                nlopt_set_ftol_abs(/*opt=*/opt, /*tol=*/control.ftol_abs);
            }
            if (control.xtol_abs > 0) {
                nlopt_set_xtol_abs1(/*opt=*/opt, /*tol=*/control.xtol_abs);
            }
            if (control.maxtime > 0) {
                nlopt_set_maxtime(/*opt=*/opt, /*maxtime=*/control.maxtime);
            }
            if (control.population > 0) {
                nlopt_set_population(/*opt=*/opt, /*pop=*/control.population);
            }
            if (control.initial_step > 0) {
                nlopt_set_initial_step1(/*opt=*/opt, /*dx=*/control.initial_step);
            }
            if (control.stopval != 0.0) {
                nlopt_set_stopval(/*opt=*/opt, /*stopval=*/control.stopval);
            }
            if (!control.x_weights.empty()) {
                if (control.x_weights.size() != static_cast<size_t>(task.params.numb_free)) {
                    throw std::invalid_argument("Error: control.x_weights length must equal number of free parameters.");
                }
                nlopt_set_x_weights(/*opt=*/opt, /*w=*/control.x_weights.data());
            }
            if (control.vector_storage > 0) {
                nlopt_set_vector_storage(/*opt=*/opt, /*M=*/control.vector_storage);
            }
            for (const auto& kv : control.nlopt_params) {
                nlopt_set_param(/*opt=*/opt, /*name=*/kv.first.c_str(), /*val=*/kv.second);
            }

            double minf = 0.0;
            // 🔥 启动探索！这行代码会内部循环调用 nll 几百上千次
            nlopt_result nlopt_res = nlopt_optimize(
                /*opt=*/opt, /*x=*/x0.data(), /*minf=*/&minf
            );
            res.n_evals = static_cast<int>(nlopt_get_numevals(opt));
            res.optimum_value = minf;
            res.result_message = nlopt_result_message(nlopt_res);
            res.stop_reason = nlopt_stop_reason(nlopt_res);

            if (nlopt_res < 0) {
                #pragma omp critical
                std::cerr << "\n[NLOPT Error] Subject " << task.subid 
                          << " failed to start optimization. NLopt Code: " 
                          << nlopt_res 
                          << " (Check if maxeval>0 and initial parameters "
                          << "are strictly inside bounds).\n";
            }
            
            // ==========================================================
            // C. 提取与封装拟合结果
            // ==========================================================
            res.status = nlopt_res;
            res.logL = -minf; // 最小化负对数似然 = 最大化对数似然

            // 计算 AIC 和 BIC
            double N = 0.0;
            for (const auto& dim_mat : task.freq.freq_mat) {
                for (const auto& row : dim_mat) {
                    for (double v : row) {
                        N += v;
                    }
                }
            }
            res.aic = 2.0 * task.params.numb_free - 2.0 * res.logL;
            res.bic = (N > 0.0) ? 
                      (task.params.numb_free * std::log(N) - 2.0 * res.logL) : 
                      0.0;

            // 将一维的极值 x0 还原回带有层级结构的字典
            auto best_p = task.params.flat;
            size_t x_idx = 0;
            for (const auto& name : task.params.name_free) {
                size_t param_len = task.params.structured.free.at(name).size();
                for (size_t k = 0; k < param_len; ++k) {
                    best_p[name][k] = x0[x_idx++];
                }
            }
            
            // 修正显示 bug：由于我们在计算概率前强制排序，优化器收敛到的是任意一种排列
            // 所以在此处需要对最佳参数进行排序，以确保返回给用户的是真实的有意义的顺序
            if (best_p.count("c_conf")) {
                std::sort(best_p["c_conf"].begin(), best_p["c_conf"].end());
            }
            if (best_p.count("d") && best_p.count("sort_d") && !best_p["sort_d"].empty() && best_p["sort_d"][0] != 0.0) {
                std::sort(best_p["d"].rbegin(), best_p["d"].rend());
            }
            
            res.best_params = best_p;
            
            // C API 需要手动释放优化器内存
            if (local_opt != NULL) nlopt_destroy(local_opt);
            nlopt_destroy(opt);

        } catch (const std::exception& e) {
            // 如果某个被试数据极端导致拟合崩溃，不会带崩其他线程
            // 加入终端警告，绝不让错误被默默吞噬！
            #pragma omp critical
            {
                std::cerr << "\n[NLOPT Error] Subject " << task.subid 
                          << " fitting failed: " << e.what() << "\n";
            }
            res.status = -1; 
            res.n_evals = 0;
            res.result_message = e.what();
            res.stop_reason = "exception";
        }

        // 将该被试的结果安全地放入预分配的槽位
        results[i] = res;
        if (control.print_level > 0) {
            ui::progress_advance(1);
        }
    }
    if (control.print_level > 0 && n_tasks > 0) {
        ui::progress_finish();
    }
    return results;
}
