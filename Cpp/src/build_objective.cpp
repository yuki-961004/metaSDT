#include "../include/build_objective.hpp"
#include "../include/criterion_likelihood.hpp"
#include "../include/data_info.hpp"
#include "../include/matrix_mult.hpp"
#include "../include/matrix_prob.hpp"
#include "../include/model_sdt.hpp"

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

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
    DataInfoResult info = data_info(/*df=*/df, /*colnames=*/colnames);

    if (!info.colnames.count("stim") || !info.colnames.count("resp")) {
        throw std::invalid_argument(
            "Error: Missing critical columns 'stim' or 'resp'."
        );
    }

    std::string col_stim = info.colnames.at("stim");
    std::string col_resp = info.colnames.at("resp");

    std::string col_conf = "";
    if (info.colnames.count("conf")) {
        col_conf = info.colnames.at("conf");
    }

    std::string col_diff = "";
    if (info.colnames.count("diff")) {
        col_diff = info.colnames.at("diff");
    }

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

    const auto& full_stim = df.at(col_stim);
    const auto& full_resp = df.at(col_resp);
    const std::vector<double>* full_conf_ptr = nullptr;
    const std::vector<double>* full_diff_ptr = nullptr;

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

    if (!col_diff.empty() && df.count(col_diff)) {
        full_diff_ptr = &df.at(col_diff);
    }

    std::vector<SubjectFitTask> tasks;

    for (const auto& kv : info.subjects) {
        double subid = kv.first;
        const auto& subj = kv.second;

        auto create_task = [&](const std::vector<int>& row_indices,
                               const std::string& cond_name) {
            std::vector<double> sub_stim;
            std::vector<double> sub_resp;
            std::vector<double> sub_conf;
            std::vector<double> sub_diff;

            sub_stim.reserve(row_indices.size());
            sub_resp.reserve(row_indices.size());

            if (full_conf_ptr != nullptr) {
                sub_conf.reserve(row_indices.size());
            }
            if (full_diff_ptr != nullptr) {
                sub_diff.reserve(row_indices.size());
            }

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

            size_t n_diffs = 1;
            if (full_diff_ptr != nullptr && !sub_diff.empty()) {
                std::set<double> diff_set(sub_diff.begin(), sub_diff.end());
                n_diffs = diff_set.size();
            }

            ParamGroup subj_user_params = user_params;
            if (!subj_user_params.free.count("d") &&
                !subj_user_params.fixed.count("d") &&
                !subj_user_params.constant.count("d")) {
                subj_user_params.free["d"] = {1.5};
            }

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

            expand_param(/*m=*/subj_user_params.free, /*key=*/"d");
            expand_param(/*m=*/subj_user_params.fixed, /*key=*/"d");
            expand_param(/*m=*/subj_user_params.constant, /*key=*/"d");

            ModifiedParamsResult subj_params = modify_params(
                /*user_params=*/subj_user_params,
                /*custom_lower=*/custom_lower,
                /*custom_upper=*/custom_upper
            );

            MatrixFreq freq_obj = matrix_freq(
                /*stim=*/sub_stim,
                /*resp=*/sub_resp,
                /*conf=*/full_conf_ptr != nullptr ? &sub_conf : nullptr,
                /*diff=*/full_diff_ptr != nullptr ? &sub_diff : nullptr,
                /*std_params=*/&subj_params.flat
            );

            SubjectFitTask task;
            task.subid = subid;
            task.cond = cond_name;
            task.model = model_name;
            task.freq = freq_obj;
            task.params = subj_params;
            task.prior = modify_prior(user_priors, subj_params, apply_priors);
            tasks.push_back(task);
        };

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

double nll(unsigned n, const double* x, double* grad, void* f_data) {
    try {
        SubjectFitTask* task = static_cast<SubjectFitTask*>(f_data);

        auto std_params = task->params.flat;
        size_t x_idx = 0;
        std::vector<double> free_params_vec;

        for (const auto& name : task->params.name_free) {
            size_t param_len = task->params.structured.free.at(name).size();
            for (size_t i = 0; i < param_len; ++i) {
                double val = x[x_idx++];
                std_params[name][i] = val;
                free_params_vec.push_back(val);
            }
        }

        auto it_c_conf = std_params.find("c_conf");
        if (it_c_conf != std_params.end() && !it_c_conf->second.empty()) {
            std::sort(it_c_conf->second.begin(), it_c_conf->second.end());
        }

        std::vector<std::vector<double>> cdf_n;
        std::vector<std::vector<double>> cdf_s;

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

        auto mult = matrix_mult<double>(
            /*freq_mat=*/task->freq.freq_mat,
            /*prob_mat=*/prob.prob_mat,
            /*std_params=*/std_params
        );

        auto loss = criterion_likelihood<double>(
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
        double log_prior = task->prior.evaluate<double>(eigen_free_params);
        return loss.nll - log_prior;
    } catch (const std::exception& e) {
        static thread_local bool error_printed = false;
        if (!error_printed) {
#pragma omp critical
            {
                std::cerr << "\n[Fatal NLL Error] " << e.what() << "\n";
            }
            error_printed = true;
        }
        (void)n;
        (void)grad;
        return 1e10;
    }
}
