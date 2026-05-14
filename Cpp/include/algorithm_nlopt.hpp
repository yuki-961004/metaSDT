#pragma once

#include "estimate_mle.hpp"

#include <nlopt.hpp>

#include <vector>

// ============================================================================
// NLopt shared algorithm utilities
// ============================================================================
// This module serves estimate_mle and estimate_map.
// Future estimate_mcmc should use a different module (algorithm_stan).

// Clip initial values to stay strictly inside optimization bounds.
void sanitize_initial_point(
    std::vector<double>& x0,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double epsilon = 1e-4
);

// Build and configure a ready-to-use nlopt optimizer instance.
nlopt::opt create_nlopt_optimizer(
    const NLoptControl& control,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds
);

