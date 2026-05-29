#include "../include/shell_run_m.hpp"

#include "../include/model_sdt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

/* ========================================================================== *
 *                              Internal Helpers                              *
 * ========================================================================== */

double scalar_param(
    const std::unordered_map<std::string, std::vector<double>>& params,
    const std::string& name
) {
    const auto it = params.find(name);
    if (it == params.end() || it->second.empty()) {
        throw std::invalid_argument("shell_run_m missing parameter: " + name);
    }
    return it->second[0];
}

std::vector<double> vector_param(
    const std::unordered_map<std::string, std::vector<double>>& params,
    const std::string& name
) {
    const auto it = params.find(name);
    if (it == params.end() || it->second.empty()) {
        throw std::invalid_argument("shell_run_m missing parameter: " + name);
    }
    return it->second;
}

double normal_density(double x, double mean, double sd) {
    if (sd <= 0.0) {
        throw std::invalid_argument("SDT standard deviations must be positive.");
    }

    const double z = (x - mean) / sd;
    const double norm = sd * std::sqrt(2.0 * M_PI);
    return std::exp(-0.5 * z * z) / norm;
}

std::vector<double> sorted_positive_conf(
    const std::unordered_map<std::string, std::vector<double>>& params
) {
    const auto it = params.find("c_conf");
    if (it == params.end() || it->second.empty()) {
        return {};
    }

    std::vector<double> out = it->second;
    std::sort(out.begin(), out.end());
    return out;
}

int confidence_level(
    double evidence,
    double c_resp,
    const std::vector<double>& c_conf
) {
    if (c_conf.empty()) {
        return 1;
    }

    const double distance = std::abs(evidence - c_resp);
    int level = 1;

    for (double boundary : c_conf) {
        if (distance >= boundary) {
            ++level;
        }
    }

    return level;
}

std::vector<double> default_xlim(
    const std::unordered_map<std::string, std::vector<double>>& params,
    const std::vector<double>& criteria
) {
    const std::vector<double> d = vector_param(params, "d");
    const double sd_noise = scalar_param(params, "sd_noise");
    const double sd_signal = scalar_param(params, "sd_signal");
    const double mu_noise = -d[0] / 2.0;
    const double mu_signal = d[0] / 2.0;

    double lower = std::min(
        mu_noise - 4.0 * sd_noise,
        mu_signal - 4.0 * sd_signal
    );
    double upper = std::max(
        mu_noise + 4.0 * sd_noise,
        mu_signal + 4.0 * sd_signal
    );

    for (double criterion : criteria) {
        lower = std::min(lower, criterion - 0.5);
        upper = std::max(upper, criterion + 0.5);
    }

    return {lower, upper};
}

ShellRunMDensity build_density(
    const std::unordered_map<std::string, std::vector<double>>& params,
    const std::vector<double>& criteria,
    const ShellRunMOptions& option
) {
    const std::vector<double> d = vector_param(params, "d");
    const double sd_noise = scalar_param(params, "sd_noise");
    const double sd_signal = scalar_param(params, "sd_signal");
    const double mu_noise = -d[0] / 2.0;
    const double mu_signal = d[0] / 2.0;
    const int n_points = std::max(option.density_points, 2);

    std::vector<double> limits = option.has_xlim
        ? option.xlim
        : default_xlim(params, criteria);

    if (limits.size() != 2 || limits[0] >= limits[1]) {
        throw std::invalid_argument(
            "option['xlim'] must contain increasing lower and upper limits."
        );
    }

    ShellRunMDensity density;
    density.x.reserve(static_cast<std::size_t>(n_points));
    density.noise.reserve(static_cast<std::size_t>(n_points));
    density.signal.reserve(static_cast<std::size_t>(n_points));

    const double step = (limits[1] - limits[0]) /
        static_cast<double>(n_points - 1);

    for (int i = 0; i < n_points; ++i) {
        const double x = limits[0] + step * static_cast<double>(i);
        density.x.push_back(x);
        density.noise.push_back(normal_density(x, mu_noise, sd_noise));
        density.signal.push_back(normal_density(x, mu_signal, sd_signal));
    }

    return density;
}

} // namespace

/* ========================================================================== *
 *                            Main Simulation Shell                           *
 * ========================================================================== */

ShellRunMResult shell_run_m(
    const ParamGroup& params,
    const std::string& model,
    const ShellRunMOptions& option
) {
    if (model != "sdt") {
        throw std::invalid_argument("shell_run_m currently supports only 'sdt'.");
    }
    if (option.n <= 0) {
        throw std::invalid_argument("option['n'] must be a positive integer.");
    }

    const ModifiedParamsResult modified = modify_params(params);
    const std::unordered_map<std::string, std::vector<double>>& flat =
        modified.flat;

    const std::vector<double> d = vector_param(flat, "d");
    const double c_resp = scalar_param(flat, "c_resp");
    const double sd_noise = scalar_param(flat, "sd_noise");
    const double sd_signal = scalar_param(flat, "sd_signal");
    const std::vector<double> c_conf = sorted_positive_conf(flat);

    ModelSDT<double> model_sdt(flat);
    const std::vector<double> criteria = model_sdt.get_criteria();

    ShellRunMResult result;
    result.model = model;
    result.params = flat;
    result.criteria = criteria;
    result.seed = option.has_seed ? option.seed : 1004;
    result.density = build_density(flat, criteria, option);

    result.data.trial.reserve(static_cast<std::size_t>(option.n));
    result.data.stim.reserve(static_cast<std::size_t>(option.n));
    result.data.resp.reserve(static_cast<std::size_t>(option.n));
    result.data.conf.reserve(static_cast<std::size_t>(option.n));
    result.data.diff.reserve(static_cast<std::size_t>(option.n));
    result.data.evidence.reserve(static_cast<std::size_t>(option.n));

    std::mt19937 rng(static_cast<std::mt19937::result_type>(result.seed));
    std::bernoulli_distribution stim_dist(0.5);
    std::uniform_int_distribution<int> diff_dist(
        0,
        static_cast<int>(d.size()) - 1
    );

    for (int i = 0; i < option.n; ++i) {
        const int diff_index = diff_dist(rng);
        const int stim_value = stim_dist(rng) ? 1 : 0;
        const double current_d = d[static_cast<std::size_t>(diff_index)];
        const double mean = stim_value == 1
            ? current_d / 2.0
            : -current_d / 2.0;
        const double sd = stim_value == 1 ? sd_signal : sd_noise;

        std::normal_distribution<double> evidence_dist(mean, sd);
        const double evidence = evidence_dist(rng);
        const int resp_value = evidence > c_resp ? 1 : 0;
        const int conf_value = confidence_level(evidence, c_resp, c_conf);

        result.data.trial.push_back(i + 1);
        result.data.stim.push_back(stim_value);
        result.data.resp.push_back(resp_value);
        result.data.conf.push_back(conf_value);
        result.data.diff.push_back(diff_index + 1);
        result.data.evidence.push_back(evidence);
    }

    return result;
}
