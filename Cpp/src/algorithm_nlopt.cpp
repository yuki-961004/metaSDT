#include "../include/algorithm_nlopt.hpp"
#include "../include/criterion_likelihood.hpp"
#include "../include/matrix_mult.hpp"
#include "../include/matrix_prob.hpp"
#include "../include/model_sdt.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

/* ========================================================================== *
 *                              NLopt Adapter                                 *
 * ========================================================================== */

namespace NLoptAdapter {

double criterion(unsigned n, const double* x, double* grad, void* f_data) {
    try {
        // NLopt 只递交自由参数, 这里像拆箱一样取回完整拟合任务.
        SubjectFitTask* task = static_cast<SubjectFitTask*>(f_data);

        // 从模板参数图出发, 用 x 覆盖自由参数位置.
        auto std_params = task->params.flat;
        size_t x_idx = 0;
        std::vector<double> free_params_vec;

        // 循环顺序必须和 name_free 一致, 这样 x 和边界才一一对应.
        for (const auto& name : task->params.name_free) {
            size_t param_len = task->params.structured.free.at(name).size();
            for (size_t i = 0; i < param_len; ++i) {
                double val = x[x_idx++];
                std_params[name][i] = val;
                free_params_vec.push_back(val);
            }
        }

        // c_conf 是有序阈值, 排序像整理刻度尺, 避免阈值交叉.
        auto it_c_conf = std_params.find("c_conf");
        if (it_c_conf != std_params.end() && !it_c_conf->second.empty()) {
            std::sort(it_c_conf->second.begin(), it_c_conf->second.end());
        }

        std::vector<std::vector<double>> cdf_n;
        std::vector<std::vector<double>> cdf_s;

        // 当前只支持 SDT 模型, 其他模型名必须明确报错.
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
            /*cdf_noise=*/cdf_n,
            /*cdf_signal=*/cdf_s,
            /*std_params=*/std_params
        );

        const auto mult = matrix_mult<double>(
            /*freq_mat=*/task->freq.freq_mat,
            /*prob_mat=*/prob.prob_mat,
            /*std_params=*/std_params
        );

        const auto loss = criterion_likelihood<double>(
            /*mult_mat=*/mult,
            /*freq_mat=*/task->freq.freq_mat,
            /*k=*/task->params.numb_free,
            /*free_params=*/free_params_vec,
            /*std_params=*/std_params
        );

        Eigen::VectorXd eigen_free_params(free_params_vec.size());
        for (size_t i = 0; i < free_params_vec.size(); ++i) {
            eigen_free_params(static_cast<Eigen::Index>(i)) =
                free_params_vec[i];
        }

        // NLopt 做最小化, 所以这里返回负对数后验.
        const double log_prior =
            task->prior.evaluate<double>(eigen_free_params);
        return loss.nll - log_prior;
    } catch (const std::exception& e) {
        static thread_local bool error_printed = false;
        if (!error_printed) {
#pragma omp critical
            {
                std::cerr << "\n[Fatal NLopt Criterion Error] "
                          << e.what() << "\n";
            }
            error_printed = true;
        }

        // NLopt callback 固定带这些参数, 出错时显式标记未使用.
        (void)n;
        (void)grad;

        // 大惩罚值像路障, 引导优化器离开非法参数区域.
        return 1e10;
    }
}

void sanitize_initial_point(
    std::vector<double>& x0,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double epsilon
) {
    for (size_t j = 0; j < x0.size(); ++j) {
        if (x0[j] <= lower_bounds[j]) {
            x0[j] = lower_bounds[j] + epsilon;
        }
        if (x0[j] >= upper_bounds[j]) {
            x0[j] = upper_bounds[j] - epsilon;
        }
    }
}

nlopt::opt build_optimizer(
    const NLoptControl& control,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds
) {
    const unsigned int n_params = static_cast<unsigned int>(
        lower_bounds.size()
    );

    nlopt::opt opt(control.algorithm.c_str(), n_params);
    opt.set_lower_bounds(lower_bounds);
    opt.set_upper_bounds(upper_bounds);

    if (control.algorithm.find("MLSL") != std::string::npos ||
        control.algorithm.find("AUGLAG") != std::string::npos) {
        nlopt::opt local_opt(control.local_algorithm.c_str(), n_params);
        local_opt.set_xtol_rel(control.xtol_rel);

        if (control.ftol_rel > 0) {
            local_opt.set_ftol_rel(control.ftol_rel);
        }
        if (control.ftol_abs > 0) {
            local_opt.set_ftol_abs(control.ftol_abs);
        }
        if (control.xtol_abs > 0) {
            local_opt.set_xtol_abs(control.xtol_abs);
        }

        opt.set_local_optimizer(local_opt);
    }

    opt.set_xtol_rel(control.xtol_rel);
    opt.set_maxeval(control.maxeval);

    if (control.ftol_rel > 0) {
        opt.set_ftol_rel(control.ftol_rel);
    }
    if (control.ftol_abs > 0) {
        opt.set_ftol_abs(control.ftol_abs);
    }
    if (control.xtol_abs > 0) {
        opt.set_xtol_abs(control.xtol_abs);
    }
    if (control.maxtime > 0) {
        opt.set_maxtime(control.maxtime);
    }
    if (control.population > 0) {
        opt.set_population(control.population);
    }
    if (control.initial_step > 0) {
        opt.set_initial_step(control.initial_step);
    }
    if (control.stopval != 0.0) {
        opt.set_stopval(control.stopval);
    }

    if (!control.x_weights.empty()) {
        if (control.x_weights.size() != lower_bounds.size()) {
            throw std::invalid_argument(
                "Error: control.x_weights length must equal number of "
                "free parameters."
            );
        }
        opt.set_x_weights(control.x_weights);
    }

    if (control.vector_storage > 0) {
        opt.set_vector_storage(control.vector_storage);
    }

    for (const auto& kv : control.nlopt_params) {
        opt.set_param(kv.first.c_str(), kv.second);
    }

    return opt;
}

} // namespace NLoptAdapter
