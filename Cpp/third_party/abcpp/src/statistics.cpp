#include "abcpp/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace abcpp {

bool is_finite(double value) {
    return std::isfinite(value);
}

double median(std::vector<double> values) {
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value) { return !std::isfinite(value); }
        ),
        values.end()
    );

    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double quantile_type7(std::vector<double> values, double probability) {
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value) { return !std::isfinite(value); }
        ),
        values.end()
    );

    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (probability <= 0.0) {
        return *std::min_element(values.begin(), values.end());
    }
    if (probability >= 1.0) {
        return *std::max_element(values.begin(), values.end());
    }

    std::sort(values.begin(), values.end());
    const double h = 1.0 + (values.size() - 1.0) * probability;
    const double lo = std::floor(h);
    const double hi = std::ceil(h);
    const std::size_t lo_index = static_cast<std::size_t>(lo) - 1;
    const std::size_t hi_index = static_cast<std::size_t>(hi) - 1;

    if (lo_index == hi_index) {
        return values[lo_index];
    }

    const double weight = h - lo;
    return values[lo_index] + weight * (values[hi_index] - values[lo_index]);
}

double mad(const std::vector<double>& values) {
    const double center = median(values);
    if (!std::isfinite(center)) {
        return 0.0;
    }

    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double value : values) {
        if (std::isfinite(value)) {
            deviations.push_back(std::fabs(value - center));
        }
    }

    const double raw_mad = median(deviations);
    if (!std::isfinite(raw_mad)) {
        return 0.0;
    }

    return 1.4826 * raw_mad;
}

double mean(const std::vector<double>& values) {
    double total = 0.0;
    std::size_t count = 0;

    for (double value : values) {
        if (std::isfinite(value)) {
            total += value;
            ++count;
        }
    }

    if (count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return total / static_cast<double>(count);
}

double weighted_mean(
    const std::vector<double>& values,
    const std::vector<double>& weights
) {
    if (values.size() != weights.size()) {
        throw std::invalid_argument("Weighted mean inputs have wrong length.");
    }

    double total = 0.0;
    double weight_total = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::isfinite(values[i]) && std::isfinite(weights[i])) {
            total += values[i] * weights[i];
            weight_total += weights[i];
        }
    }

    if (weight_total <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return total / weight_total;
}

double sample_sd(const std::vector<double>& values) {
    const double mu = mean(values);
    if (!std::isfinite(mu)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double total = 0.0;
    std::size_t count = 0;
    for (double value : values) {
        if (std::isfinite(value)) {
            const double diff = value - mu;
            total += diff * diff;
            ++count;
        }
    }

    if (count < 2) {
        return 0.0;
    }
    return std::sqrt(total / static_cast<double>(count - 1));
}

std::vector<double> column_means(const Matrix& matrix) {
    std::vector<double> out(matrix.cols(), 0.0);
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
        out[col] = mean(matrix.col(col));
    }
    return out;
}

std::vector<double> column_mads(
    const Matrix& matrix,
    const std::vector<bool>& keep_rows
) {
    std::vector<double> out(matrix.cols(), 0.0);
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
        std::vector<double> values;
        for (std::size_t row = 0; row < matrix.rows(); ++row) {
            if (keep_rows.empty() || keep_rows[row]) {
                values.push_back(matrix(row, col));
            }
        }
        out[col] = mad(values);
    }
    return out;
}

std::vector<double> column_weighted_sse(
    const Matrix& matrix,
    const std::vector<double>& weights
) {
    if (matrix.rows() != weights.size()) {
        throw std::invalid_argument("Weighted SSE inputs have wrong length.");
    }

    std::vector<double> out(matrix.cols(), 0.0);
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
        for (std::size_t row = 0; row < matrix.rows(); ++row) {
            const double value = matrix(row, col);
            out[col] += weights[row] * value * value;
        }
    }
    return out;
}

double kernel_mode(
    const std::vector<double>& values,
    const std::vector<double>& weights
) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double lo = *std::min_element(values.begin(), values.end());
    const double hi = *std::max_element(values.begin(), values.end());
    if (lo == hi) {
        return lo;
    }

    const double bandwidth = std::max(
        sample_sd(values) * std::pow(values.size(), -0.2),
        (hi - lo) / 100.0
    );

    double best_x = values[0];
    double best_density = -1.0;
    for (std::size_t i = 0; i <= 200; ++i) {
        const double x = lo + (hi - lo) * static_cast<double>(i) / 200.0;
        double density = 0.0;
        for (std::size_t j = 0; j < values.size(); ++j) {
            const double z = (x - values[j]) / bandwidth;
            const double weight = weights.empty() ? 1.0 : weights[j];
            density += weight * std::exp(-0.5 * z * z);
        }

        if (density > best_density) {
            best_density = density;
            best_x = x;
        }
    }

    return best_x;
}

}  // namespace abcpp
