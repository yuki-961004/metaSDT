#include "../include/modify_control.hpp"

#include <algorithm>
#include <cctype>

namespace {
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}
}

NLoptControl modify_control(
    const NLoptControl& input, const std::string& estimator
) {
    NLoptControl out = input;

    if (out.algorithm.empty()) out.algorithm = "LN_BOBYQA";
    if (out.local_algorithm.empty()) out.local_algorithm = "LN_BOBYQA";

    if (out.xtol_rel <= 0.0) out.xtol_rel = 1e-6;
    if (out.maxeval <= 0) out.maxeval = 10000;
    if (out.print_level < 0) out.print_level = 0;
    if (out.threads < 0) out.threads = 0;
    if (out.seed < 0) out.seed = 1004;
    if (out.progress.empty()) out.progress = "dynamic";
    out.progress = to_lower(out.progress);
    if (out.progress != "auto" && out.progress != "dynamic" &&
        out.progress != "line" && out.progress != "silent") {
        out.progress = "auto";
    }
    if (out.progress_refresh_ms <= 0) out.progress_refresh_ms = 100;
    if (out.progress_line_interval_sec <= 0.0) {
        out.progress_line_interval_sec = 2.0;
    }
    if (out.progress_line_interval_pct <= 0.0) {
        out.progress_line_interval_pct = 5.0;
    }

    const std::string mode = to_lower(estimator);
    if (mode == "map") {
        if (out.em_max_iter <= 0) out.em_max_iter = 100;
        if (out.em_tol <= 0.0) out.em_tol = 1e-3;
        if (out.em_patience < 0) out.em_patience = 0;
    }

    return out;
}
