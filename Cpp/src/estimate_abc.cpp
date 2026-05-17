#include "../include/estimate_abc.hpp"
#include "../include/matrix_prob.hpp"
#include "../include/model_sdt.hpp"

#include <abcpp/abc.hpp>
#include <abcpp/options.hpp>
#include <abcpp/summary.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

std::vector<double> flatten_freq(const MatrixFreq& freq) {
    std::vector<double> out;
    for (const auto& dim : freq.freq_mat) {
        for (const auto& row : dim) {
            out.insert(out.end(), row.begin(), row.end());
        }
    }
    return out;
}

template <typename T>
std::vector<double> flatten_prob_counts(
    const MatrixProb<T>& prob,
    double total_trials
) {
    std::vector<double> out;
    for (const auto& dim : prob.prob_mat) {
        for (const auto& row : dim) {
            for (const T value : row) {
                out.push_back(std::llround(static_cast<double>(value) * total_trials));
            }
        }
    }
    return out;
}

double sum_values(const std::vector<double>& values) {
    double out = 0.0;
    for (const double value : values) {
        out += value;
    }
    return out;
}

int infer_effective_ncomp(const MatrixFreq& freq) {
    std::size_t out = 0;
    for (const auto& dim : freq.freq_mat) {
        for (const auto& row : dim) {
            if (!row.empty()) {
                out += row.size() - 1;
            }
        }
    }
    return static_cast<int>(out);
}

std::vector<std::string> ordered_base_names(
    const std::unordered_map<std::string, std::vector<double>>& params
) {
    std::vector<std::string> names;
    names.reserve(params.size());
    for (const auto& kv : params) {
        names.push_back(kv.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::vector<std::string> flatten_parameter_names(
    const ModifiedParamsResult& params
) {
    std::vector<std::string> out;
    for (const auto& name : params.name_free) {
        const auto& values = params.structured.free.at(name);
        if (values.size() <= 1) {
            out.push_back(name);
        } else {
            for (std::size_t i = 0; i < values.size(); ++i) {
                out.push_back(name + "_" + std::to_string(i + 1));
            }
        }
    }
    return out;
}

std::vector<double> flatten_parameter_values(
    const std::unordered_map<std::string, std::vector<double>>& params,
    const std::vector<std::string>& base_names
) {
    std::vector<double> out;
    for (const auto& name : base_names) {
        const auto it = params.find(name);
        if (it == params.end()) {
            throw std::invalid_argument(
                "Every ABC parameter sample must contain parameter '" + name + "'."
            );
        }
        out.insert(out.end(), it->second.begin(), it->second.end());
    }
    return out;
}

double prior_arg(
    const UserPrior& prior,
    const std::vector<std::string>& keys,
    double fallback
) {
    for (const auto& key : keys) {
        const auto it = prior.args.find(key);
        if (it != prior.args.end()) {
            return it->second;
        }
    }
    return fallback;
}

std::string lower_string(std::string x) {
    std::transform(x.begin(), x.end(), x.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return x;
}

double sample_from_prior(
    const std::string& name,
    double initial,
    const std::unordered_map<std::string, UserPrior>& priors,
    std::mt19937& rng
) {
    const auto it = priors.find(name);
    if (it == priors.end()) {
        const double sd = std::max(std::abs(initial) * 0.5, 1.0);
        std::normal_distribution<double> dist(initial, sd);
        return dist(rng);
    }

    const UserPrior& prior = it->second;
    const std::string type = lower_string(prior.type);

    if (type == "normal" || type == "norm") {
        const double mean = prior_arg(prior, {"mean", "mu", "location", "param1"}, initial);
        const double sd = std::max(
            prior_arg(prior, {"sd", "sigma", "scale", "param2"}, 1.0),
            1e-12
        );
        std::normal_distribution<double> dist(mean, sd);
        return dist(rng);
    }
    if (type == "uniform" || type == "unif") {
        const double lower = prior_arg(prior, {"min", "lower", "param1"}, initial - 1.0);
        const double upper = prior_arg(prior, {"max", "upper", "param2"}, initial + 1.0);
        std::uniform_real_distribution<double> dist(
            std::min(lower, upper),
            std::max(lower, upper)
        );
        return dist(rng);
    }
    if (type == "lognormal" || type == "lnorm") {
        const double meanlog = prior_arg(prior, {"mean", "meanlog", "mu", "param1"}, 0.0);
        const double sdlog = std::max(
            prior_arg(prior, {"sd", "sdlog", "sigma", "param2"}, 1.0),
            1e-12
        );
        std::lognormal_distribution<double> dist(meanlog, sdlog);
        return dist(rng);
    }
    if (type == "cauchy") {
        const double location = prior_arg(prior, {"mean", "location", "param1"}, initial);
        const double scale = std::max(
            prior_arg(prior, {"sd", "scale", "param2"}, 1.0),
            1e-12
        );
        std::cauchy_distribution<double> dist(location, scale);
        double value = dist(rng);
        if (!std::isfinite(value)) {
            value = location;
        }
        return value;
    }
    if (type == "beta") {
        const double a = std::max(
            prior_arg(prior, {"shape1", "alpha", "param1"}, 1.0),
            1e-12
        );
        const double b = std::max(
            prior_arg(prior, {"shape2", "beta", "param2"}, 1.0),
            1e-12
        );
        std::gamma_distribution<double> ga(a, 1.0);
        std::gamma_distribution<double> gb(b, 1.0);
        const double x = ga(rng);
        const double y = gb(rng);
        return (x + y > 0.0) ? x / (x + y) : 0.5;
    }
    if (type == "exponential" || type == "exp") {
        const double rate = std::max(
            prior_arg(prior, {"rate", "lambda", "param1"}, 1.0),
            1e-12
        );
        std::exponential_distribution<double> dist(rate);
        return dist(rng);
    }

    const double sd = std::max(std::abs(initial) * 0.5, 1.0);
    std::normal_distribution<double> dist(initial, sd);
    return dist(rng);
}

std::unordered_map<std::string, UserPrior> merged_priors(
    const std::unordered_map<std::string, UserPrior>& user_priors
) {
    auto out = default_priors();
    for (const auto& kv : user_priors) {
        out[kv.first] = kv.second;
    }
    return out;
}

std::vector<std::unordered_map<std::string, std::vector<double>>> draw_param_samples(
    const SubjectFitTask& task,
    int n_samples,
    const std::unordered_map<std::string, UserPrior>& priors,
    std::mt19937& rng
) {
    std::vector<std::unordered_map<std::string, std::vector<double>>> out;
    out.reserve(static_cast<std::size_t>(n_samples));

    for (int s = 0; s < n_samples; ++s) {
        auto sample = task.params.flat;
        for (const auto& name : task.params.name_free) {
            const auto& initial_values = task.params.structured.free.at(name);
            std::vector<double> values;
            values.reserve(initial_values.size());
            for (const double initial : initial_values) {
                values.push_back(sample_from_prior(name, initial, priors, rng));
            }
            if (name == "c_conf") {
                std::sort(values.begin(), values.end());
            }
            sample[name] = values;
        }
        out.push_back(std::move(sample));
    }

    return out;
}

abcpp::AbcOptions make_options(const ABCControl& control, int n_comp) {
    abcpp::AbcOptions options;
    options.tol = control.tol;
    options.method = abcpp::parse_method(control.method);
    options.kernel = abcpp::parse_kernel(control.kernel);
    options.hcorr = control.hcorr;
    options.seed = control.seed;
    options.reduction.method = abcpp::parse_reduction(control.reduction);
    options.reduction.ncomp = static_cast<std::size_t>(
        std::max(n_comp, 0)
    );
    return options;
}

ABCSummaryStats convert_summary(const abcpp::SummaryColumn& col) {
    ABCSummaryStats out;
    out.min = col.min;
    out.q_lower = col.q_lower;
    out.median = col.median;
    out.mean = col.mean;
    out.mode = col.mode;
    out.q_upper = col.q_upper;
    out.max = col.max;
    out.sd = col.sd;
    return out;
}

} // namespace

std::vector<SubjectABCResult> estimate_abc(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const ABCControl& control,
    const std::unordered_map<std::string, UserPrior>& user_priors
) {
    const int n_samples = control.samples > 0 ? control.samples : 1000;
    const auto prior_map = merged_priors(user_priors);

    std::vector<SubjectFitTask> tasks = build_fit_tasks(
        df,
        colnames,
        user_params,
        model_name
    );

    std::vector<SubjectABCResult> results;
    results.reserve(tasks.size());

    for (std::size_t task_index = 0; task_index < tasks.size(); ++task_index) {
        const auto& task = tasks[task_index];
        SubjectABCResult out;
        out.subid = task.subid;
        out.cond = task.cond;
        const std::vector<std::string> base_param_names = task.params.name_free;
        const std::vector<std::string> flat_param_names =
            flatten_parameter_names(task.params);
        if (flat_param_names.empty()) {
            throw std::invalid_argument(
                "estimate_abc requires at least one free parameter in params."
            );
        }
        out.parameter_names = flat_param_names;

        try {
            std::mt19937 rng(
                static_cast<unsigned int>(
                    control.seed + static_cast<unsigned int>(task_index)
                )
            );
            const auto param_samples = draw_param_samples(
                task,
                n_samples,
                prior_map,
                rng
            );

            abcpp::Matrix param_matrix(
                param_samples.size(),
                flat_param_names.size()
            );
            for (std::size_t r = 0; r < param_samples.size(); ++r) {
                const std::vector<double> values =
                    flatten_parameter_values(param_samples[r], base_param_names);
                for (std::size_t c = 0; c < values.size(); ++c) {
                    param_matrix(r, c) = values[c];
                }
            }

            const std::vector<double> target = flatten_freq(task.freq);
            const double total_trials = sum_values(target);
            if (target.empty() || total_trials <= 0.0) {
                throw std::invalid_argument("ABC target frequency matrix is empty.");
            }
            const int effective_n_comp = control.n_comp > 0
                ? control.n_comp
                : infer_effective_ncomp(task.freq);
            out.n_comp_used = effective_n_comp;

            abcpp::Matrix sumstat_matrix(param_samples.size(), target.size());
            for (std::size_t i = 0; i < param_samples.size(); ++i) {
                ModelSDT<double> sdt(param_samples[i]);
                MatrixProb<double> prob = matrix_prob<double>(
                    sdt.cdf_noise(),
                    sdt.cdf_signal(),
                    param_samples[i]
                );
                const std::vector<double> sim_counts =
                    flatten_prob_counts(prob, total_trials);
                if (sim_counts.size() != target.size()) {
                    throw std::invalid_argument(
                        "Simulated ABC summary width does not match target width."
                    );
                }
                for (std::size_t j = 0; j < sim_counts.size(); ++j) {
                    sumstat_matrix(i, j) = sim_counts[j];
                }
            }

            abcpp::AbcResult abc_res = abcpp::abc(
                target,
                param_matrix,
                sumstat_matrix,
                make_options(control, effective_n_comp)
            );
            abc_res.parameter_names = flat_param_names;

            const abcpp::SummaryResult summary = abcpp::summary(abc_res);
            out.summary.reserve(summary.columns.size());
            for (const auto& col : summary.columns) {
                out.summary.push_back(convert_summary(col));
            }
            out.accepted_distances = abc_res.distances;
            out.accepted_indices = abc_res.accepted_indices;
            out.status = abc_res.status == "ok" ? 0 : -1;
            out.message = abc_res.message;
        } catch (const std::exception& e) {
            out.status = -1;
            out.message = e.what();
        }

        results.push_back(out);
    }

    return results;
}
