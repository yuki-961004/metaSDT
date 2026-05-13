#include "../include/estimate_map.hpp"
#include "../include/build_objective.hpp"
#include "../include/model_sdt.hpp"
#include "../include/matrix_prob.hpp"
#include "../include/matrix_mult.hpp"
#include "../include/criterion_likelihood.hpp"
#include "../include/modify_control.hpp"
#include "../include/progress_bar.hpp"

#include <nlopt.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
struct CondRunResult {
    std::vector<SubjectFitResult> results;
    int iter_used = 0;
    std::string stop_reason = "not_started";
};

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

std::vector<double> flatten_free_from_params(
    const SubjectFitTask& task,
    const std::unordered_map<std::string, std::vector<double>>& params
) {
    std::vector<double> out;
    out.reserve(task.params.numb_free);
    for (const auto& name : task.params.name_free) {
        const auto it = params.find(name);
        if (it == params.end()) continue;
        out.insert(out.end(), it->second.begin(), it->second.end());
    }
    return out;
}

void update_task_start_from_best(SubjectFitTask& task, const SubjectFitResult& res) {
    for (const auto& name : task.params.name_free) {
        auto it_best = res.best_params.find(name);
        if (it_best == res.best_params.end()) continue;
        task.params.structured.free[name] = it_best->second;
        task.params.flat[name] = it_best->second;
    }
}

std::vector<SubjectFitResult> optimize_tasks(
    std::vector<SubjectFitTask>& tasks,
    const NLoptControl& control,
    const std::string& progress_title = ""
) {
    std::vector<SubjectFitResult> results(tasks.size());

    const int n_tasks = static_cast<int>(tasks.size());
    if (control.print_level > 0 && n_tasks > 0 && !progress_title.empty()) {
        ui::ProgressOptions popts;
        popts.mode = control.progress;
        popts.refresh_ms = control.progress_refresh_ms;
        popts.line_interval_sec = control.progress_line_interval_sec;
        popts.line_interval_pct = control.progress_line_interval_pct;
        ui::progress_start(static_cast<std::size_t>(n_tasks), progress_title, control.progress_refresh_ms, popts);
    }
    #pragma omp parallel for
    for (int i = 0; i < n_tasks; ++i) {
        auto& task = tasks[i];
        SubjectFitResult res;
        res.subid = task.subid;
        res.cond = task.cond;
        res.best_params = task.params.flat;

        std::vector<double> x0;
        x0.reserve(task.params.numb_free);
        for (const auto& name : task.params.name_free) {
            const auto& vals = task.params.structured.free.at(name);
            for (double v : vals) x0.push_back(v);
        }

        for (size_t j = 0; j < x0.size(); ++j) {
            if (x0[j] <= task.params.lower_bounds[j]) x0[j] = task.params.lower_bounds[j] + 1e-4;
            if (x0[j] >= task.params.upper_bounds[j]) x0[j] = task.params.upper_bounds[j] - 1e-4;
        }

        try {
            nlopt_algorithm algo = nlopt_algorithm_from_string(control.algorithm.c_str());
            if (static_cast<int>(algo) < 0) {
                throw std::invalid_argument("Error: Invalid NLopt algorithm name: '" + control.algorithm + "'");
            }

            nlopt_opt opt = nlopt_create(algo, task.params.numb_free);
            nlopt_set_lower_bounds(opt, task.params.lower_bounds.data());
            nlopt_set_upper_bounds(opt, task.params.upper_bounds.data());

            nlopt_opt local_opt = NULL;
            if (control.algorithm.find("MLSL") != std::string::npos ||
                control.algorithm.find("AUGLAG") != std::string::npos) {
                nlopt_algorithm local_algo = nlopt_algorithm_from_string(control.local_algorithm.c_str());
                if (static_cast<int>(local_algo) < 0) {
                    throw std::invalid_argument("Error: Invalid local NLopt algorithm name: '" + control.local_algorithm + "'");
                }
                local_opt = nlopt_create(local_algo, task.params.numb_free);
                nlopt_set_xtol_rel(local_opt, control.xtol_rel);
                if (control.ftol_rel > 0) nlopt_set_ftol_rel(local_opt, control.ftol_rel);
                if (control.ftol_abs > 0) nlopt_set_ftol_abs(local_opt, control.ftol_abs);
                if (control.xtol_abs > 0) nlopt_set_xtol_abs1(local_opt, control.xtol_abs);
                nlopt_set_local_optimizer(opt, local_opt);
            }

            nlopt_set_min_objective(opt, nll, &task);
            nlopt_set_xtol_rel(opt, control.xtol_rel);
            nlopt_set_maxeval(opt, control.maxeval);
            if (control.ftol_rel > 0) nlopt_set_ftol_rel(opt, control.ftol_rel);
            if (control.ftol_abs > 0) nlopt_set_ftol_abs(opt, control.ftol_abs);
            if (control.xtol_abs > 0) nlopt_set_xtol_abs1(opt, control.xtol_abs);
            if (control.maxtime > 0) nlopt_set_maxtime(opt, control.maxtime);
            if (control.population > 0) nlopt_set_population(opt, control.population);
            if (control.initial_step > 0) nlopt_set_initial_step1(opt, control.initial_step);
            if (control.stopval != 0.0) nlopt_set_stopval(opt, control.stopval);
            if (!control.x_weights.empty()) {
                if (control.x_weights.size() != static_cast<size_t>(task.params.numb_free)) {
                    throw std::invalid_argument("Error: control.x_weights length must equal number of free parameters.");
                }
                nlopt_set_x_weights(opt, control.x_weights.data());
            }
            if (control.vector_storage > 0) {
                nlopt_set_vector_storage(opt, control.vector_storage);
            }
            for (const auto& kv : control.nlopt_params) {
                nlopt_set_param(opt, kv.first.c_str(), kv.second);
            }

            double minf = 0.0;
            nlopt_result nlopt_res = nlopt_optimize(opt, x0.data(), &minf);
            res.status = nlopt_res;
            res.n_evals = static_cast<int>(nlopt_get_numevals(opt));
            res.optimum_value = minf;
            res.result_message = nlopt_result_message(nlopt_res);
            res.stop_reason = nlopt_stop_reason(nlopt_res);

            auto best_p = task.params.flat;
            size_t x_idx = 0;
            for (const auto& name : task.params.name_free) {
                size_t param_len = task.params.structured.free.at(name).size();
                for (size_t k = 0; k < param_len; ++k) {
                    best_p[name][k] = x0[x_idx++];
                }
            }
            if (best_p.count("c_conf")) std::sort(best_p["c_conf"].begin(), best_p["c_conf"].end());
            if (best_p.count("d") && best_p.count("sort_d") && !best_p["sort_d"].empty() && best_p["sort_d"][0] != 0.0) {
                std::sort(best_p["d"].rbegin(), best_p["d"].rend());
            }

            std::vector<double> free_params_vec = flatten_free_from_params(task, best_p);
            auto std_params = best_p;

            std::vector<std::vector<double>> cdf_n;
            std::vector<std::vector<double>> cdf_s;
            if (task.model == "sdt") {
                ModelSDT<double> model_obj(std_params);
                cdf_n = model_obj.cdf_noise();
                cdf_s = model_obj.cdf_signal();
            } else {
                throw std::invalid_argument("Error: Unknown model name '" + task.model + "'.");
            }

            MatrixProb<double> prob = matrix_prob<double>(cdf_n, cdf_s, std_params);
            auto mult = matrix_mult<double>(task.freq.freq_mat, prob.prob_mat, std_params);
            auto loss = criterion_likelihood<double>(mult, task.freq.freq_mat, task.params.numb_free, free_params_vec, std_params);

            Eigen::VectorXd eigen_free_params(free_params_vec.size());
            for (size_t p_idx = 0; p_idx < free_params_vec.size(); ++p_idx) {
                eigen_free_params(p_idx) = free_params_vec[p_idx];
            }
            double log_prior = task.prior.evaluate<double>(eigen_free_params);

            res.logL = loss.logL;
            res.logPrior = log_prior;
            res.logPost = res.logL + res.logPrior;

            double N = 0.0;
            for (const auto& dim_mat : task.freq.freq_mat) {
                for (const auto& row : dim_mat) {
                    for (double v : row) N += v;
                }
            }
            res.aic = 2.0 * task.params.numb_free - 2.0 * res.logL;
            res.bic = (N > 0.0) ? (task.params.numb_free * std::log(N) - 2.0 * res.logL) : 0.0;
            res.best_params = best_p;

            if (local_opt != NULL) nlopt_destroy(local_opt);
            nlopt_destroy(opt);
        } catch (const std::exception& e) {
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
        if (control.print_level > 0 && !progress_title.empty()) {
            ui::progress_advance(1);
        }
    }
    if (control.print_level > 0 && n_tasks > 0 && !progress_title.empty()) {
        ui::progress_finish();
    }

    return results;
}

CondRunResult run_em_for_condition(
    std::vector<SubjectFitTask> tasks,
    const NLoptControl& control,
    const std::string& condition_col_name
) {
    CondRunResult out;
    if (tasks.empty()) {
        out.stop_reason = "empty";
        return out;
    }

    if (control.em_init_mle) {
        std::vector<SubjectFitTask> mle_tasks = tasks;
        for (auto& t : mle_tasks) t.prior = CriterionPrior();
        std::vector<SubjectFitResult> mle_res = optimize_tasks(mle_tasks, control);
        for (size_t i = 0; i < tasks.size() && i < mle_res.size(); ++i) {
            update_task_start_from_best(tasks[i], mle_res[i]);
        }
    }

    std::vector<SubjectFitResult> best_results = optimize_tasks(tasks, control);
    double best_sum_logpost = 0.0;
    for (const auto& r : best_results) best_sum_logpost += r.logPost;
    double prev_sum_logpost = best_sum_logpost;
    bool use_patience = control.em_patience > 0;
    int patience_left = control.em_patience;

    out.results = best_results;
    out.stop_reason = "init_only";

    if (control.print_level > 0 && control.em_max_iter > 0) {
        std::string title = "MAP";
        if (!condition_col_name.empty()) title += " " + condition_col_name;
        if (!tasks.empty() && !tasks[0].cond.empty()) title += " [" + tasks[0].cond + "]";
        ui::ProgressOptions popts;
        popts.mode = control.progress;
        popts.refresh_ms = control.progress_refresh_ms;
        popts.line_interval_sec = control.progress_line_interval_sec;
        popts.line_interval_pct = control.progress_line_interval_pct;
        ui::progress_start(static_cast<std::size_t>(control.em_max_iter), title, control.progress_refresh_ms, popts);
    }

    for (int iter = 0; iter < control.em_max_iter; ++iter) {
        out.iter_used = iter + 1;
        if (control.print_level > 0) {
            ui::progress_set(static_cast<std::size_t>(out.iter_used));
        }

        std::unordered_map<size_t, std::vector<std::vector<double>>> grouped_free;
        for (size_t i = 0; i < tasks.size(); ++i) {
            std::vector<double> fp = flatten_free_from_params(tasks[i], best_results[i].best_params);
            grouped_free[fp.size()].push_back(fp);
            update_task_start_from_best(tasks[i], best_results[i]);
        }

        for (auto& task : tasks) {
            const size_t k = static_cast<size_t>(task.params.numb_free);
            auto it = grouped_free.find(k);
            if (it != grouped_free.end() && it->second.size() >= 2) {
                task.prior.update(it->second);
            }
        }

        std::vector<SubjectFitResult> current = optimize_tasks(tasks, control);
        double sum_logpost = 0.0;
        for (const auto& r : current) sum_logpost += r.logPost;
        const double delta = sum_logpost - prev_sum_logpost;

        if (std::abs(delta) <= control.em_tol) {
            out.results = current;
            out.stop_reason = "em_tol";
            if (control.print_level > 0 && control.em_max_iter > 0) ui::progress_finish();
            return out;
        }

        if (sum_logpost > best_sum_logpost) {
            best_sum_logpost = sum_logpost;
            best_results = current;
            out.results = current;
            if (use_patience) {
                patience_left = std::min(control.em_patience, patience_left + 1);
            }
        } else {
            if (use_patience) {
                patience_left -= 1;
            }
        }

        prev_sum_logpost = sum_logpost;
        if (use_patience && patience_left <= 0) {
            out.stop_reason = "em_patience";
            if (control.print_level > 0 && control.em_max_iter > 0) ui::progress_finish();
            return out;
        }
    }

    out.stop_reason = "em_max_iter";
    if (control.print_level > 0 && control.em_max_iter > 0) ui::progress_finish();
    return out;
}

} // namespace

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
    if (control.threads > 0) {
        omp_set_num_threads(control.threads);
    }
#endif
    if (control.seed >= 0) {
        nlopt_srand(static_cast<unsigned long>(control.seed));
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

    if (tasks.empty()) return {};

    std::unordered_map<std::string, std::vector<SubjectFitTask>> tasks_by_cond;
    for (const auto& t : tasks) {
        tasks_by_cond[t.cond].push_back(t);
    }

    std::string condition_col_name;
    auto it_cond = colnames.find("condition");
    if (it_cond != colnames.end()) {
        condition_col_name = it_cond->second;
    }

    std::vector<SubjectFitResult> all_results;
    int total_iter = 0;
    for (auto& kv : tasks_by_cond) {
        CondRunResult cond_res = run_em_for_condition(kv.second, control, condition_col_name);
        total_iter += cond_res.iter_used;
        all_results.insert(all_results.end(), cond_res.results.begin(), cond_res.results.end());
        if (em_iterations_by_cond != nullptr) {
            (*em_iterations_by_cond)[kv.first] = cond_res.iter_used;
        }
        if (em_stop_reason_by_cond != nullptr) {
            (*em_stop_reason_by_cond)[kv.first] = cond_res.stop_reason;
        }
    }

    if (em_iterations_used != nullptr) *em_iterations_used = total_iter;
    return all_results;
}
