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

struct CondRunResult {
    std::vector<SubjectFitResult> results;
    int iter_used = 0;
    std::string stop_reason = "not_started";
};

std::vector<double> flatten_free_from_params(
    const SubjectFitTask& task,
    const std::unordered_map<std::string, std::vector<double>>& params
) {
    std::vector<double> out;
    out.reserve(task.params.numb_free);

    for (const auto& name : task.params.name_free) {
        const auto it = params.find(name);
        if (it == params.end()) {
            continue;
        }
        out.insert(out.end(), it->second.begin(), it->second.end());
    }

    return out;
}

std::vector<int> free_param_sizes(const SubjectFitTask& task) {
    std::vector<int> sizes;
    sizes.reserve(task.params.name_free.size());

    for (const auto& name : task.params.name_free) {
        const int p_size = static_cast<int>(
            task.params.structured.free.at(name).size()
        );
        sizes.push_back(p_size);
    }

    return sizes;
}

std::vector<SubjectFitResult> em_e_step(
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
        ui::progress_start(
            static_cast<std::size_t>(n_tasks),
            progress_title,
            control.progress_refresh_ms,
            popts
        );
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
            for (const double v : vals) {
                x0.push_back(v);
            }
        }

        sanitize_initial_point(
            x0,
            task.params.lower_bounds,
            task.params.upper_bounds
        );

        try {
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
            size_t x_idx = 0;
            for (const auto& name : task.params.name_free) {
                const size_t p_len = task.params.structured.free.at(name).size();
                for (size_t k = 0; k < p_len; ++k) {
                    best_p[name][k] = x0[x_idx++];
                }
            }
            if (best_p.count("c_conf")) {
                std::sort(best_p["c_conf"].begin(), best_p["c_conf"].end());
            }
            if (best_p.count("d") && best_p.count("sort_d") &&
                !best_p["sort_d"].empty() && best_p["sort_d"][0] != 0.0) {
                std::sort(best_p["d"].rbegin(), best_p["d"].rend());
            }

            const std::vector<double> free_params_vec =
                flatten_free_from_params(task, best_p);
            Eigen::VectorXd free_params(
                static_cast<Eigen::Index>(free_params_vec.size())
            );
            for (size_t p_idx = 0; p_idx < free_params_vec.size(); ++p_idx) {
                free_params(static_cast<Eigen::Index>(p_idx)) =
                    free_params_vec[p_idx];
            }

            const CriterionPosterior posterior(
                task.freq.freq_mat,
                task.params.name_free,
                free_param_sizes(task),
                task.params.flat,
                task.prior
            );
            const double log_post = posterior.operator()<double>(free_params);

            std::vector<std::vector<double>> cdf_n;
            std::vector<std::vector<double>> cdf_s;
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
            for (const auto& dim_mat : task.freq.freq_mat) {
                for (const auto& row : dim_mat) {
                    for (const double v : row) {
                        N += v;
                    }
                }
            }
            res.aic = 2.0 * task.params.numb_free - 2.0 * res.logL;
            res.bic = (N > 0.0)
                ? (task.params.numb_free * std::log(N) - 2.0 * res.logL)
                : 0.0;
            res.best_params = best_p;
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

CondRunResult em_m_step(
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
        for (auto& t : mle_tasks) {
            t.prior = CriterionPrior();
        }
        const std::vector<SubjectFitResult> mle_res = em_e_step(
            mle_tasks,
            control
        );
        for (size_t i = 0; i < tasks.size() && i < mle_res.size(); ++i) {
            optimal(tasks[i], mle_res[i]);
        }
    }

    std::vector<SubjectFitResult> best_results = em_e_step(tasks, control);
    double best_sum_logpost = 0.0;
    for (const auto& r : best_results) {
        best_sum_logpost += r.logPost;
    }

    double prev_sum_logpost = best_sum_logpost;
    const bool use_patience = control.em_patience > 0;
    int patience_left = control.em_patience;

    out.results = best_results;
    out.stop_reason = "init_only";

    if (control.print_level > 0 && control.em_max_iter > 0) {
        std::string title = "MAP";
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

    for (int iter = 0; iter < control.em_max_iter; ++iter) {
        out.iter_used = iter + 1;
        if (control.print_level > 0) {
            ui::progress_set(static_cast<std::size_t>(out.iter_used));
        }

        std::unordered_map<size_t, std::vector<std::vector<double>>> grouped;
        for (size_t i = 0; i < tasks.size(); ++i) {
            std::vector<double> fp = flatten_free_from_params(
                tasks[i],
                best_results[i].best_params
            );
            grouped[fp.size()].push_back(fp);
            optimal(tasks[i], best_results[i]);
        }

        for (auto& task : tasks) {
            const size_t k = static_cast<size_t>(task.params.numb_free);
            const auto it = grouped.find(k);
            if (it != grouped.end() && it->second.size() >= 2) {
                task.prior.update(it->second);
            }
        }

        std::vector<SubjectFitResult> current = em_e_step(tasks, control);

        double sum_logpost = 0.0;
        for (const auto& r : current) {
            sum_logpost += r.logPost;
        }
        const double delta = sum_logpost - prev_sum_logpost;

        if (std::abs(delta) <= control.em_tol) {
            out.results = current;
            out.stop_reason = "em_tol";
            if (control.print_level > 0 && control.em_max_iter > 0) {
                ui::progress_finish();
            }
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

    if (tasks.empty()) {
        return {};
    }

    std::unordered_map<std::string, std::vector<SubjectFitTask>> tasks_by_cond;
    for (const auto& t : tasks) {
        tasks_by_cond[t.cond].push_back(t);
    }

    std::string condition_col_name;
    const auto it_cond = colnames.find("condition");
    if (it_cond != colnames.end()) {
        condition_col_name = it_cond->second;
    }

    std::vector<SubjectFitResult> all_results;
    int total_iter = 0;

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
        if (em_iterations_by_cond != nullptr) {
            (*em_iterations_by_cond)[kv.first] = cond_res.iter_used;
        }
        if (em_stop_reason_by_cond != nullptr) {
            (*em_stop_reason_by_cond)[kv.first] = cond_res.stop_reason;
        }
    }

    if (em_iterations_used != nullptr) {
        *em_iterations_used = total_iter;
    }

    return all_results;
}
