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
