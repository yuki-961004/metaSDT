#pragma once

#include <string>
#include "estimate_mle.hpp"

// Normalize control values and inject estimator-specific defaults.
NLoptControl modify_control(const NLoptControl& input, const std::string& estimator);
