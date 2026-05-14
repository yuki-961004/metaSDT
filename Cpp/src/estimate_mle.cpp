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

            auto best_p = task.params.flat;
            size_t x_idx = 0;
            for (const auto& name : task.params.name_free) {
                const size_t param_len =
                    task.params.structured.free.at(name).size();
                for (size_t k = 0; k < param_len; ++k) {
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

            res.best_params = best_p;
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "\n[NLOPT Error] Subject " << task.subid
                          << " fitting failed: " << e.what() << "\n";
            }
            if (res.status == 0) {
                res.status = -1;
            }
            if (res.result_message.empty()) {
                res.result_message = e.what();
            }
            if (res.stop_reason.empty()) {
                res.stop_reason = "exception";
            }
        }

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
