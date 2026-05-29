#include "../include/algorithm_abcpp.hpp"
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

/* ========================================================================== *
 *                              Internal Helpers                              *
 * ========================================================================== */

// 将多维观测频数矩阵展平为一维向量, 以便作为 ABC 算法的观测目标
std::vector<double> flatten_freq(const MatrixFreq& freq) {
    std::vector<double> out;
    for (const auto& dim : freq.freq_mat) {
        for (const auto& row : dim) {
            out.insert(out.end(), row.begin(), row.end());
        }
    }
    return out;
}

// 根据模型输出的概率矩阵和总试验次数, 计算期望计数值并展平为一维向量
template <typename T>
std::vector<double> flatten_prob_counts(
    const MatrixProb<T>& prob,
    double total_trials
) {
    std::vector<double> out;
    for (const auto& dim : prob.prob_mat) {
        for (const auto& row : dim) {
            for (const T value : row) {
                out.push_back(
                    std::llround(static_cast<double>(value) * total_trials)
                );
            }
        }
    }
    return out;
}

// 辅助函数: 计算向量所有元素的总和
double sum_values(const std::vector<double>& values) {
    double out = 0.0;
    for (const double value : values) {
        out += value;
    }
    return out;
}

// 根据观测矩阵的结构自动推断用于部分最小二乘 (PLS) 降维的有效主成分数量
int infer_effective_n_comp(const MatrixFreq& freq) {
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

// 将自由参数名称按照其向量长度展开为扁平的字符串列表 (例如 "d_1", "d_2")
std::vector<std::string> flatten_parameter_names(
    const ModifiedParamsResult& params
) {
    std::vector<std::string> out;
    for (const auto& name : params.name_free) {
        const std::vector<double>& values =
            params.structured.free.at(name);
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

// 根据基准名称列表, 从参数字典中提取并展平对应的数值以构造参数矩阵的行
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

// 辅助函数: 尝试从先验字典参数中获取值, 若未找到则返回后备值
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

// 将字符串转换为小写以进行宽容的模式匹配
std::string lower_string(std::string x) {
    std::transform(x.begin(), x.end(), x.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return x;
}

// 根据指定的先验分布类型从随机数生成器中抽取单个样本
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
        const double mean = prior_arg(
            prior,
            {"mean", "mu", "location", "param1"},
            initial
        );
        const double sd = std::max(
            prior_arg(prior, {"sd", "sigma", "scale", "param2"}, 1.0),
            1e-12
        );
        std::normal_distribution<double> dist(mean, sd);
        return dist(rng);
    }
    if (type == "uniform" || type == "unif") {
        const double lower = prior_arg(
            prior,
            {"min", "lower", "param1"},
            initial - 1.0
        );
        const double upper = prior_arg(
            prior,
            {"max", "upper", "param2"},
            initial + 1.0
        );
        std::uniform_real_distribution<double> dist(
            std::min(lower, upper),
            std::max(lower, upper)
        );
        return dist(rng);
    }
    if (type == "lognormal" || type == "lnorm") {
        const double meanlog = prior_arg(
            prior,
            {"mean", "meanlog", "mu", "param1"},
            0.0
        );
        const double sdlog = std::max(
            prior_arg(prior, {"sd", "sdlog", "sigma", "param2"}, 1.0),
            1e-12
        );
        std::lognormal_distribution<double> dist(meanlog, sdlog);
        return dist(rng);
    }
    if (type == "cauchy") {
        const double location = prior_arg(
            prior,
            {"mean", "location", "param1"},
            initial
        );
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

// 为给定的拟合任务批量生成基于先验分布的候选参数样本
std::vector<std::unordered_map<std::string, std::vector<double>>>
draw_param_samples(
    const SubjectFitTask& task,
    int n_samples,
    const std::unordered_map<std::string, UserPrior>& priors,
    std::mt19937& rng
) {
    std::vector<std::unordered_map<std::string, std::vector<double>>> out;
    out.reserve(static_cast<std::size_t>(n_samples));

    for (int s = 0; s < n_samples; ++s) {
        std::unordered_map<std::string, std::vector<double>> sample =
            task.params.flat;
        for (const auto& name : task.params.name_free) {
            const std::vector<double>& initial_values =
                task.params.structured.free.at(name);
            std::vector<double> values;
            values.reserve(initial_values.size());
            for (const double initial : initial_values) {
                values.push_back(
                    sample_from_prior(name, initial, priors, rng)
                );
            }
            // 确保置信度相关的标准点严格排序
            if (name == "c_conf") {
                std::sort(values.begin(), values.end());
            }
            sample[name] = values;
        }
        out.push_back(std::move(sample));
    }

    return out;
}

// 将字符串格式的数据变换方法名称转换为 abcpp 底层支持的枚举类型
std::vector<abcpp::transform> parse_transformations(
    const std::vector<std::string>& values
) {
    std::vector<abcpp::transform> out;
    out.reserve(values.size());
    for (const auto& value : values) {
        out.push_back(abcpp::parse_transform(value));
    }
    return out;
}

// 将标准库的二维向量嵌套结构转换为 abcpp 内部的矩阵对象
abcpp::Matrix to_abcpp_matrix(
    const std::vector<std::vector<double>>& values
) {
    if (values.empty()) {
        return abcpp::Matrix();
    }

    const std::size_t n_cols = values.front().size();
    abcpp::Matrix out(values.size(), n_cols);
    for (std::size_t r = 0; r < values.size(); ++r) {
        if (values[r].size() != n_cols) {
            throw std::invalid_argument(
                "ABC control logit_bounds must be a rectangular matrix."
            );
        }
        for (std::size_t c = 0; c < n_cols; ++c) {
            out(r, c) = values[r][c];
        }
    }
    return out;
}

// 提取 abcpp 矩阵的第一列并将其返回为标准的浮点数向量
std::vector<double> matrix_first_column(const abcpp::Matrix& matrix) {
    std::vector<double> out;
    out.reserve(matrix.rows());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        out.push_back(matrix.cols() == 0 ? 0.0 : matrix(row, 0));
    }
    return out;
}

// 将顶层的 ABC 控制选项映射转换为底层 abcpp 所需的计算配置对象
abcpp::AbcOptions make_options(const ABCControl& control, int n_comp) {
    abcpp::AbcOptions options;
    options.tol = control.tol;
    options.method = abcpp::parse_method(control.method);
    options.kernel = abcpp::parse_kernel(control.kernel);
    options.hcorr = control.hcorr;
    options.transformations = parse_transformations(control.transf);
    if (!options.transformations.empty()) {
        options.transf = options.transformations.front();
    }
    options.logit_bounds = to_abcpp_matrix(control.logit_bounds);
    options.subset = control.subset;
    options.prior_weights = control.prior_weights;
    options.seed = control.seed;
    options.nnet.numnet = control.nnet.numnet;
    options.nnet.sizenet = control.nnet.sizenet;
    options.nnet.lambda = control.nnet.lambda;
    options.nnet.maxit = control.nnet.maxit;
    options.nnet.rang = control.nnet.rang;
    options.nnet.abstol = control.nnet.abstol;
    options.nnet.reltol = control.nnet.reltol;
    options.nnet.verbose = control.nnet.verbose;
    options.nnet.skip = control.nnet.skip;
    options.reduction.method = abcpp::parse_reduction(control.reduction);
    options.reduction.n_comp = static_cast<std::size_t>(
        std::max(n_comp, 0)
    );
    return options;
}

// 将底层 abcpp 返回的统计摘要信息转换为上层模块使用的公共摘要结构
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

namespace abcppAdapter {

std::unordered_map<std::string, UserPrior> merged_priors(
    const std::unordered_map<std::string, UserPrior>& user_priors
) {
    std::unordered_map<std::string, UserPrior> out = default_priors();
    for (const auto& kv : user_priors) {
        out[kv.first] = kv.second;
    }
    return out;
}

SubjectABCResult run_subject_abc(
    const SubjectFitTask& task,
    const ABCControl& control,
    const std::unordered_map<std::string, UserPrior>& prior_map,
    int task_index
) {
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

    std::mt19937 rng(
        static_cast<unsigned int>(
            control.seed + static_cast<unsigned int>(task_index)
        )
    );
    const std::vector<std::unordered_map<std::string, std::vector<double>>>
        param_samples = draw_param_samples(
            task,
            control.samples,
            prior_map,
            rng
        );

    abcpp::Matrix param_matrix(param_samples.size(), flat_param_names.size());
    for (std::size_t r = 0; r < param_samples.size(); ++r) {
        const std::vector<double> values = flatten_parameter_values(
            param_samples[r],
            base_param_names
        );
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
        : infer_effective_n_comp(task.freq);
    out.n_comp_used = effective_n_comp;

    abcpp::Matrix sumstat_matrix(param_samples.size(), target.size());
    for (std::size_t i = 0; i < param_samples.size(); ++i) {
        ModelSDT<double> sdt(param_samples[i]);
        const MatrixProb<double> prob = matrix_prob<double>(
            sdt.cdf_noise(),
            sdt.cdf_signal(),
            param_samples[i]
        );
        const std::vector<double> sim_counts = flatten_prob_counts(
            prob,
            total_trials
        );
        if (sim_counts.size() != target.size()) {
            throw std::invalid_argument(
                "Simulated ABC summary width does not match target width."
            );
        }
        for (std::size_t j = 0; j < sim_counts.size(); ++j) {
            sumstat_matrix(i, j) = sim_counts[j];
        }
    }

    abcpp::AbcResult abc_res = abcpp::fit(
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
    out.accepted_weights = matrix_first_column(abc_res.weights);
    out.status = abc_res.status == "ok" ? 0 : -1;
    out.message = abc_res.message;

    return out;
}

} // namespace abcppAdapter
