#pragma once

#include "estimate_mle.hpp"

#include <nlopt.hpp>

#include <vector>

/* ========================================================================== *
 *                           NLopt Adapter API                                *
 * ========================================================================== */

namespace NLoptAdapter {

// Calculate the criterion minimized by NLopt.
// 这个函数像 NLopt 和模型之间的翻译员, 把自由参数向量还原为模型参数.
double criterion(
    unsigned n,
    const double* x,
    double* grad,
    void* f_data
);

// Clip initial values to stay strictly inside optimization bounds.
// 这一步像把起跑点推回跑道内, 避免优化器从边界外直接摔倒.
void sanitize_initial_point(
    std::vector<double>& x0,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double epsilon = 1e-4
);

// Build and configure a ready-to-use nlopt optimizer instance.
// 这个函数只装配优化器, 真正的目标函数由 criterion 提供.
nlopt::opt build_optimizer(
    const NLoptControl& control,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds
);

} // namespace NLoptAdapter

