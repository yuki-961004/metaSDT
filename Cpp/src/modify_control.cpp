#include "../include/modify_control.hpp"

#include <algorithm>
#include <cctype>

namespace {

/* ========================================================================== *
 *                              Internal Helpers                              *
 * ========================================================================== */

// 将字符串转换为小写, 以便进行不区分大小写的匹配
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string normalize_stan_algorithm(std::string algorithm) {
    algorithm = to_lower(algorithm);
    std::replace(algorithm.begin(), algorithm.end(), '-', '_');
    std::replace(algorithm.begin(), algorithm.end(), '.', '_');

    if (algorithm.empty()) {
        return "nuts";
    }

    // Stan 用户常见命名会包含 unit_e 前缀, 这里统一到项目内部主引擎.
    if (algorithm == "nuts" ||
        algorithm == "unit_e_nuts" ||
        algorithm == "adapt_unit_e_nuts") {
        return "nuts";
    }

    // hmc 在 Stan 语境下通常对应固定轨迹的 static HMC.
    if (algorithm == "hmc" ||
        algorithm == "static" ||
        algorithm == "static_hmc" ||
        algorithm == "unit_e_hmc" ||
        algorithm == "unit_e_static_hmc" ||
        algorithm == "adapt_unit_e_static_hmc") {
        return "static_hmc";
    }

    return "nuts";
}

} // namespace

/* ========================================================================== *
 *                    Normalize Control and Fill Default Values               *
 * ========================================================================== */

NLoptControl modify_control(
    const NLoptControl& input,
    const std::string& estimator
) {
    // 在副本上进行操作, 以避免就地修改调用者提供的值
    NLoptControl out = input;

/* -------------------------------------------------------------------------- *
 *                 1) Default values for optimizer settings                   *
 * -------------------------------------------------------------------------- */

    // 如果缺少算法设置, 则使用稳健的无导数默认算法
    if (out.algorithm.empty()) {
        out.algorithm = "LN_BOBYQA";
    }
    if (out.local_algorithm.empty()) {
        out.local_algorithm = "LN_BOBYQA";
    }

    // 无效或非正的数值将被替换为保守的默认值
    if (out.xtol_rel <= 0.0) {
        out.xtol_rel = 1e-6;
    }
    if (out.maxeval <= 0) {
        out.maxeval = 10000;
    }
    if (out.print_level < 0) {
        out.print_level = 0;
    }
    if (out.threads < 0) {
        out.threads = 0;
    }
    if (out.seed < 0) {
        out.seed = 1004;
    }

/* -------------------------------------------------------------------------- *
 *                  2) Progress display mode normalization                    *
 * -------------------------------------------------------------------------- */

    // 当未提供进度显示模式时, 默认使用 "dynamic"(动态模式)
    if (out.progress.empty()) {
        out.progress = "dynamic";
    }
    out.progress = to_lower(out.progress);

    // 仅允许已知的进度显示模式; 否则回退为 "auto"(自动模式)
    if (out.progress != "auto" &&
        out.progress != "dynamic" &&
        out.progress != "line" &&
        out.progress != "silent") {
        out.progress = "auto";
    }

    // 进度显示的时间控制值必须为正数才有意义
    if (out.progress_refresh_ms <= 0) {
        out.progress_refresh_ms = 100;
    }
    if (out.progress_line_interval_sec <= 0.0) {
        out.progress_line_interval_sec = 2.0;
    }
    if (out.progress_line_interval_pct <= 0.0) {
        out.progress_line_interval_pct = 5.0;
    }

/* -------------------------------------------------------------------------- *
 *                  3) Estimator-specific parameter fixes                     *
 * -------------------------------------------------------------------------- */

    // 在进行分支匹配前, 先对估计器标签进行规范化
    const std::string mode = to_lower(estimator);

    // MAP 估计使用 EM 控制参数; 在此处修复无效的值
    if (mode == "map") {
        if (out.em_max_iter <= 0) {
            out.em_max_iter = 100;
        }
        if (out.em_tol <= 0.0) {
            out.em_tol = 1e-3;
        }
        if (out.em_patience < 0) {
            out.em_patience = 0;
        }
    }

    // 返回规范化后的控制参数对象, 以便下游代码安全使用
    return out;
}

/* ========================================================================== *
 *                    Normalize Stan Control and Defaults                     *
 * ========================================================================== */

StanControl modify_control(
    const StanControl& input,
    const std::string& estimator
) {
    // 在副本上修正参数, 避免修改调用者原始配置
    StanControl out = input;

    const std::string mode = to_lower(estimator);

    // 仅在 MCMC 模式下做采样器专属修正, 让调用意图保持清晰
    if (mode != "mcmc") {
        return out;
    }

    // 采样器算法目前支持固定轨迹 HMC 和自适应树深度 NUTS
    // 将 Stan 风格算法名折叠到当前实现的两个主采样引擎.
    out.algorithm = normalize_stan_algorithm(out.algorithm);

    // chains 至少为 1, 否则没有可运行的 Markov chain
    if (out.chains <= 0) {
        out.chains = 4;
    }

    // warmup 允许为 0, 但负数没有统计意义
    if (out.warmup < 0) {
        out.warmup = 0;
    }

    // samples 至少为 1, 否则无法返回后验样本
    if (out.samples <= 0) {
        out.samples = 1000;
    }

    // thin 至少为 1, 防止取模逻辑除以零
    if (out.thin <= 0) {
        out.thin = 1;
    }

    // leapfrog 至少执行一步, 否则 proposal 不会移动
    if (out.leapfrog_steps <= 0) {
        out.leapfrog_steps = 10;
    }

    // NUTS 树深度过大会指数级增加 leapfrog 次数, 因此限制范围
    if (out.max_tree_depth <= 0) {
        out.max_tree_depth = 8;
    }
    if (out.max_tree_depth > 20) {
        out.max_tree_depth = 20;
    }

    // 步长必须严格为正, 非法输入回退为保守默认值
    if (out.step_size <= 0.0) {
        out.step_size = 0.05;
    }
    if (out.min_step_size <= 0.0) {
        out.min_step_size = 1e-6;
    }
    if (out.max_step_size <= out.min_step_size) {
        out.max_step_size = 1.0;
    }
    out.step_size = std::max(
        out.min_step_size,
        std::min(out.step_size, out.max_step_size)
    );

    // target_accept 限制在合理开区间内, 避免 warmup 更新失控
    if (out.target_accept <= 0.0 || out.target_accept >= 1.0) {
        out.target_accept = 0.80;
    }

    // NUTS 通过能量差识别发散轨迹, 非正值回退到经典阈值
    if (out.max_delta_energy <= 0.0) {
        out.max_delta_energy = 1000.0;
    }

    // jitter 不能为负, 0 表示所有链从同一点开始
    if (out.initial_jitter < 0.0) {
        out.initial_jitter = 0.0;
    }

    // seed 负值时回退到项目默认种子
    if (out.seed < 0) {
        out.seed = 1004;
    }

    // 打印级别和线程数使用非负约束
    if (out.print_level < 0) {
        out.print_level = 0;
    }
    if (out.threads < 0) {
        out.threads = 0;
    }

    // 进度模式沿用 MLE/MAP 的四种模式
    if (out.progress.empty()) {
        out.progress = "dynamic";
    }
    out.progress = to_lower(out.progress);
    if (out.progress != "auto" &&
        out.progress != "dynamic" &&
        out.progress != "line" &&
        out.progress != "silent") {
        out.progress = "auto";
    }

    // 进度刷新参数必须为正, 否则恢复为稳定默认值
    if (out.progress_refresh_ms <= 0) {
        out.progress_refresh_ms = 100;
    }
    if (out.progress_line_interval_sec <= 0.0) {
        out.progress_line_interval_sec = 2.0;
    }
    if (out.progress_line_interval_pct <= 0.0) {
        out.progress_line_interval_pct = 5.0;
    }

    return out;
}
