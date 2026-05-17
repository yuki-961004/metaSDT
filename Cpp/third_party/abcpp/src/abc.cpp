#include "abcpp/abc.hpp"

#include "abcpp/linear_algebra.hpp"
#include "abcpp/reduction.hpp"
#include "abcpp/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace abcpp {

namespace {

constexpr double kTiny = 1e-12;

std::vector<Transform> normalized_transforms(
    const AbcOptions& options,
    std::size_t numparam
) {
    std::vector<Transform> transforms = options.transformations;
    if (transforms.empty()) {
        transforms.push_back(Transform::None);
    }

    if (options.method == Method::Rejection) {
        return std::vector<Transform>(numparam, Transform::None);
    }

    if (transforms.size() == 1 && numparam > 1) {
        transforms.assign(numparam, transforms.front());
    }

    if (transforms.size() != numparam) {
        throw std::invalid_argument(
            "Number of transformations must match parameters."
        );
    }

    return transforms;
}

bool row_is_finite(const Matrix& matrix, std::size_t row) {
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
        if (!is_finite(matrix(row, col))) {
            return false;
        }
    }
    return true;
}

std::vector<bool> build_good_rows(
    const Matrix& param,
    const Matrix& sumstat,
    const std::vector<bool>& subset
) {
    std::vector<bool> keep(sumstat.rows(), true);
    for (std::size_t row = 0; row < sumstat.rows(); ++row) {
        // 这里同时检查参数和统计量, 避免回归阶段接收到非有限值.
        keep[row] = row_is_finite(sumstat, row) && row_is_finite(param, row);
        if (!subset.empty()) {
            // subset 是用户显式筛选条件, FALSE 的行直接排除.
            keep[row] = keep[row] && subset[row];
        }
    }
    return keep;
}

Matrix scale_sumstat(
    const Matrix& sumstat,
    const std::vector<bool>& good_rows,
    std::vector<double>& target
) {
    Matrix scaled = sumstat;
    for (std::size_t col = 0; col < sumstat.cols(); ++col) {
        std::vector<double> ref;
        for (std::size_t row = 0; row < sumstat.rows(); ++row) {
            if (good_rows[row]) {
                ref.push_back(sumstat(row, col));
            }
        }

        const double scale = mad(ref);
        if (scale == 0.0) {
            continue;
        }

        // R abc::normalise 只除以 MAD, 不做均值中心化.
        for (std::size_t row = 0; row < sumstat.rows(); ++row) {
            scaled(row, col) = sumstat(row, col) / scale;
        }
        target[col] /= scale;
    }
    return scaled;
}

std::vector<double> compute_distances(
    const Matrix& scaled_sumstat,
    const std::vector<double>& target,
    const std::vector<bool>& good_rows
) {
    std::vector<double> dist(scaled_sumstat.rows(), 0.0);
    double max_good = 0.0;

    for (std::size_t row = 0; row < scaled_sumstat.rows(); ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < scaled_sumstat.cols(); ++col) {
            const double diff = scaled_sumstat(row, col) - target[col];
            sum += diff * diff;
        }
        dist[row] = std::sqrt(sum);
        if (good_rows[row]) {
            max_good = std::max(max_good, dist[row]);
        }
    }

    for (std::size_t row = 0; row < scaled_sumstat.rows(); ++row) {
        if (!good_rows[row]) {
            // 保留 R abc 的思路, 坏行距离放大到不会被 tol 选中.
            dist[row] = std::floor(max_good + 10.0);
        }
    }

    return dist;
}

std::vector<bool> select_region(
    const std::vector<double>& distances,
    const std::vector<bool>& good_rows,
    double tol,
    Kernel kernel
) {
    if (tol <= 0.0 || tol > 1.0) {
        throw std::invalid_argument("tol must be in (0, 1].");
    }

    const std::size_t n = distances.size();
    std::size_t good_count = 0;
    for (bool keep : good_rows) {
        if (keep) {
            ++good_count;
        }
    }

    std::size_t nacc = static_cast<std::size_t>(
        std::ceil(static_cast<double>(good_count) * tol)
    );
    nacc = std::max<std::size_t>(1, std::min(nacc, good_count));

    std::vector<double> sorted;
    sorted.reserve(good_count);
    for (std::size_t row = 0; row < n; ++row) {
        if (good_rows[row]) {
            sorted.push_back(distances[row]);
        }
    }
    std::sort(sorted.begin(), sorted.end());
    const double cutoff = sorted[nacc - 1];

    std::vector<bool> region(n, false);
    std::size_t accepted = 0;
    for (std::size_t row = 0; row < n; ++row) {
        // ties 按原始行顺序截断, 这复刻 R 代码中的 cumsum 行为.
        if (good_rows[row] && distances[row] <= cutoff && accepted < nacc) {
            region[row] = true;
            ++accepted;
        }
    }

    if (kernel == Kernel::Gaussian) {
        for (std::size_t row = 0; row < n; ++row) {
            // gaussian kernel 在 R abc 中会使用全部可用模拟样本.
            region[row] = good_rows[row];
        }
    }

    return region;
}

double distance_cutoff(
    const std::vector<double>& distances,
    const std::vector<bool>& good_rows,
    double tol
) {
    std::size_t good_count = 0;
    for (bool keep : good_rows) {
        if (keep) {
            ++good_count;
        }
    }

    std::size_t nacc = static_cast<std::size_t>(
        std::ceil(static_cast<double>(good_count) * tol)
    );
    nacc = std::max<std::size_t>(
        1,
        std::min<std::size_t>(nacc, good_count)
    );

    std::vector<double> sorted;
    sorted.reserve(good_count);
    for (std::size_t row = 0; row < distances.size(); ++row) {
        if (good_rows[row]) {
            sorted.push_back(distances[row]);
        }
    }
    std::sort(sorted.begin(), sorted.end());
    return sorted[nacc - 1];
}

std::vector<double> build_weights(
    const std::vector<double>& distances,
    const std::vector<std::size_t>& accepted_rows,
    double cutoff,
    Kernel kernel
) {
    std::vector<double> weights(accepted_rows.size(), 1.0);
    const double denom = std::max(cutoff, kTiny);

    for (std::size_t i = 0; i < accepted_rows.size(); ++i) {
        const double ratio = distances[accepted_rows[i]] / denom;
        switch (kernel) {
        case Kernel::Epanechnikov:
            weights[i] = 1.0 - ratio * ratio;
            break;
        case Kernel::Rectangular:
            weights[i] = ratio;
            break;
        case Kernel::Gaussian: {
            const double sigma = std::max(cutoff / 2.0, kTiny);
            const double z = distances[accepted_rows[i]] / sigma;
            weights[i] = std::exp(-0.5 * z * z) /
                std::sqrt(2.0 * 3.14159265358979323846);
            break;
        }
        case Kernel::Triangular:
            weights[i] = 1.0 - std::fabs(ratio);
            break;
        case Kernel::Biweight:
            weights[i] = (1.0 - ratio * ratio) * (1.0 - ratio * ratio);
            break;
        case Kernel::Cosine:
            weights[i] = std::cos(
                3.14159265358979323846 * 0.5 * ratio
            );
            break;
        }

        if (!std::isfinite(weights[i]) || weights[i] < 0.0) {
            weights[i] = 0.0;
        }
    }

    return weights;
}

Matrix transform_parameters(
    const Matrix& param,
    const std::vector<Transform>& transforms,
    const Matrix& logit_bounds
) {
    Matrix out = param;
    for (std::size_t col = 0; col < param.cols(); ++col) {
        if (transforms[col] == Transform::Log) {
            double replacement = std::numeric_limits<double>::infinity();
            for (std::size_t row = 0; row < param.rows(); ++row) {
                if (param(row, col) > 0.0) {
                    replacement = std::min(replacement, param(row, col));
                }
            }
            if (!std::isfinite(replacement)) {
                replacement = 1.0;
            }

            for (std::size_t row = 0; row < param.rows(); ++row) {
                const double value = param(row, col) <= 0.0
                    ? replacement
                    : param(row, col);
                out(row, col) = std::log(value);
            }
        }

        if (transforms[col] == Transform::Logit) {
            if (logit_bounds.rows() <= col || logit_bounds.cols() != 2) {
                throw std::invalid_argument("Logit bounds are missing.");
            }

            const double lower = logit_bounds(col, 0);
            const double upper = logit_bounds(col, 1);
            if (lower >= upper) {
                throw std::invalid_argument("Logit bounds are incorrect.");
            }

            double low_replacement = std::numeric_limits<double>::infinity();
            double high_replacement = -std::numeric_limits<double>::infinity();
            for (std::size_t row = 0; row < param.rows(); ++row) {
                if (param(row, col) > lower) {
                    low_replacement = std::min(
                        low_replacement,
                        param(row, col)
                    );
                }
                if (param(row, col) < upper) {
                    high_replacement = std::max(
                        high_replacement,
                        param(row, col)
                    );
                }
            }

            for (std::size_t row = 0; row < param.rows(); ++row) {
                double value = param(row, col);
                if (value <= lower) {
                    value = low_replacement;
                }
                if (value >= upper) {
                    value = high_replacement;
                }
                value = (value - lower) / (upper - lower);
                value = std::min(1.0 - kTiny, std::max(kTiny, value));
                out(row, col) = std::log(value / (1.0 - value));
            }
        }
    }
    return out;
}

void back_transform(
    Matrix& matrix,
    const std::vector<Transform>& transforms,
    const Matrix& logit_bounds
) {
    if (matrix.empty()) {
        return;
    }

    for (std::size_t col = 0; col < matrix.cols(); ++col) {
        for (std::size_t row = 0; row < matrix.rows(); ++row) {
            if (transforms[col] == Transform::Log) {
                matrix(row, col) = std::exp(matrix(row, col));
            }
            if (transforms[col] == Transform::Logit) {
                const double lower = logit_bounds(col, 0);
                const double upper = logit_bounds(col, 1);
                const double exp_value = std::exp(matrix(row, col));
                double value = exp_value / (1.0 + exp_value);
                value = value * (upper - lower) + lower;
                matrix(row, col) = value;
            }
        }
    }
}

Matrix make_target_design(const std::vector<double>& target) {
    Matrix out(1, target.size() + 1, 1.0);
    for (std::size_t col = 0; col < target.size(); ++col) {
        out(0, col + 1) = target[col];
    }
    return out;
}

Matrix accepted_matrix(
    const Matrix& matrix,
    const std::vector<std::size_t>& accepted_rows
) {
    Matrix out(accepted_rows.size(), matrix.cols(), 0.0);
    for (std::size_t row = 0; row < accepted_rows.size(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            out(row, col) = matrix(accepted_rows[row], col);
        }
    }
    return out;
}

Matrix repeat_row(const Matrix& row_matrix, std::size_t rows) {
    Matrix out(rows, row_matrix.cols(), 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < row_matrix.cols(); ++col) {
            out(row, col) = row_matrix(0, col);
        }
    }
    return out;
}

Matrix residual_matrix(const Matrix& response, const Matrix& fitted) {
    Matrix out(response.rows(), response.cols(), 0.0);
    for (std::size_t row = 0; row < response.rows(); ++row) {
        for (std::size_t col = 0; col < response.cols(); ++col) {
            out(row, col) = response(row, col) - fitted(row, col);
        }
    }
    return out;
}

void center_residuals_and_shift_prediction(
    Matrix& residuals,
    Matrix& prediction
) {
    for (std::size_t col = 0; col < residuals.cols(); ++col) {
        std::vector<double> values = residuals.col(col);
        const double residual_mean = mean(values);
        for (std::size_t row = 0; row < residuals.rows(); ++row) {
            residuals(row, col) -= residual_mean;
            prediction(row, col) += residual_mean;
        }
    }
}

Matrix log_square(const Matrix& matrix) {
    Matrix out(matrix.rows(), matrix.cols(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            const double value = std::max(matrix(row, col) *
                matrix(row, col), kTiny);
            out(row, col) = std::log(value);
        }
    }
    return out;
}

Matrix heteroscedastic_adjust(
    const Matrix& prediction,
    const Matrix& residuals,
    const Matrix& design,
    const Matrix& target_design,
    const std::vector<double>& weights,
    double ridge_lambda
) {
    const Matrix log_residuals = log_square(residuals);
    const Matrix coef = weighted_least_squares(
        design,
        log_residuals,
        weights,
        ridge_lambda
    );

    const Matrix pred_sd_row = predict(target_design, coef);
    const Matrix fitted_sd = predict(design, coef);
    Matrix out(prediction.rows(), prediction.cols(), 0.0);

    for (std::size_t row = 0; row < prediction.rows(); ++row) {
        for (std::size_t col = 0; col < prediction.cols(); ++col) {
            const double target_sd = std::sqrt(std::exp(pred_sd_row(0, col)));
            const double local_sd = std::sqrt(std::exp(fitted_sd(row, col)));
            out(row, col) = prediction(row, col) +
                target_sd * residuals(row, col) / std::max(local_sd, kTiny);
        }
    }
    return out;
}

Matrix run_loclinear(
    const Matrix& design,
    const Matrix& target_design,
    const Matrix& response,
    const std::vector<double>& weights,
    bool hcorr,
    double& aic,
    double& bic,
    Matrix& residuals
) {
    const Matrix coef = weighted_least_squares(design, response, weights, 0.0);
    const Matrix fitted = predict(design, coef);
    Matrix prediction = repeat_row(predict(target_design, coef), response.rows());
    residuals = residual_matrix(response, fitted);
    center_residuals_and_shift_prediction(residuals, prediction);

    double weight_total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weight_total <= 0.0) {
        weight_total = 1.0;
    }

    double log_sigma_total = 0.0;
    for (std::size_t col = 0; col < residuals.cols(); ++col) {
        double sigma2 = 0.0;
        for (std::size_t row = 0; row < residuals.rows(); ++row) {
            sigma2 += weights[row] * residuals(row, col) * residuals(row, col);
        }
        sigma2 /= weight_total;
        log_sigma_total += std::log(std::max(sigma2, kTiny));
    }

    aic = static_cast<double>(response.rows()) * log_sigma_total +
        2.0 * static_cast<double>(design.cols() * response.cols());
    bic = static_cast<double>(response.rows()) * log_sigma_total +
        std::log(static_cast<double>(response.rows())) *
        static_cast<double>(design.cols() * response.cols());

    if (hcorr) {
        Matrix adjusted = heteroscedastic_adjust(
            prediction,
            residuals,
            design,
            target_design,
            weights,
            0.0
        );
        residuals = residual_matrix(adjusted, prediction);
        return adjusted;
    }

    Matrix adjusted(prediction.rows(), prediction.cols(), 0.0);
    for (std::size_t row = 0; row < adjusted.rows(); ++row) {
        for (std::size_t col = 0; col < adjusted.cols(); ++col) {
            adjusted(row, col) = prediction(row, col) + residuals(row, col);
        }
    }
    return adjusted;
}

std::vector<double> param_mads(
    const Matrix& param,
    const std::vector<bool>& good_rows
) {
    std::vector<double> out(param.cols(), 1.0);
    for (std::size_t col = 0; col < param.cols(); ++col) {
        std::vector<double> values;
        for (std::size_t row = 0; row < param.rows(); ++row) {
            if (good_rows[row]) {
                values.push_back(param(row, col));
            }
        }
        out[col] = std::max(mad(values), kTiny);
    }
    return out;
}

Matrix divide_columns(
    const Matrix& matrix,
    const std::vector<double>& scales
) {
    Matrix out = matrix;
    for (std::size_t row = 0; row < out.rows(); ++row) {
        for (std::size_t col = 0; col < out.cols(); ++col) {
            out(row, col) /= scales[col];
        }
    }
    return out;
}

void multiply_columns(
    Matrix& matrix,
    const std::vector<double>& scales
) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            matrix(row, col) *= scales[col];
        }
    }
}

Matrix median_matrices(const std::vector<Matrix>& values) {
    if (values.empty()) {
        return Matrix();
    }

    Matrix out(values.front().rows(), values.front().cols(), 0.0);
    for (std::size_t row = 0; row < out.rows(); ++row) {
        for (std::size_t col = 0; col < out.cols(); ++col) {
            std::vector<double> cell;
            cell.reserve(values.size());
            for (const Matrix& matrix : values) {
                cell.push_back(matrix(row, col));
            }
            out(row, col) = median(cell);
        }
    }
    return out;
}

std::vector<double> ridge_lm_coefficients(
    const Matrix& design,
    const std::vector<double>& response,
    const std::vector<double>& weights,
    double lambda
) {
    const std::size_t rows = design.rows();
    const std::size_t predictors = design.cols() - 1;
    Matrix x_weighted(rows, predictors, 0.0);
    std::vector<double> y_weighted(rows, 0.0);

    for (std::size_t row = 0; row < rows; ++row) {
        const double sqrt_weight = std::sqrt(std::max(weights[row], 0.0));
        y_weighted[row] = sqrt_weight * response[row];
        for (std::size_t col = 0; col < predictors; ++col) {
            x_weighted(row, col) = sqrt_weight * design(row, col + 1);
        }
    }

    std::vector<double> x_mean(predictors, 0.0);
    for (std::size_t col = 0; col < predictors; ++col) {
        x_mean[col] = mean(x_weighted.col(col));
    }
    const double y_mean = mean(y_weighted);

    Matrix x_centered(rows, predictors, 0.0);
    std::vector<double> y_centered(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        y_centered[row] = y_weighted[row] - y_mean;
        for (std::size_t col = 0; col < predictors; ++col) {
            x_centered(row, col) = x_weighted(row, col) - x_mean[col];
        }
    }

    std::vector<double> x_scale(predictors, 1.0);
    for (std::size_t col = 0; col < predictors; ++col) {
        double square_total = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            square_total += x_centered(row, col) * x_centered(row, col);
        }
        const double scale = std::sqrt(square_total / rows);
        x_scale[col] = scale > kTiny ? scale : 1.0;
        for (std::size_t row = 0; row < rows; ++row) {
            x_centered(row, col) /= x_scale[col];
        }
    }

    Matrix xtx(predictors, predictors, 0.0);
    std::vector<double> xty(predictors, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t left = 0; left < predictors; ++left) {
            xty[left] += x_centered(row, left) * y_centered[row];
            for (std::size_t right = 0; right < predictors; ++right) {
                xtx(left, right) += x_centered(row, left) *
                    x_centered(row, right);
            }
        }
    }

    for (std::size_t diag = 0; diag < predictors; ++diag) {
        xtx(diag, diag) += lambda;
    }

    std::vector<double> beta_scaled = solve_linear_system(xtx, xty);
    std::vector<double> coef(predictors + 1, 0.0);
    for (std::size_t col = 0; col < predictors; ++col) {
        coef[col + 1] = beta_scaled[col] / x_scale[col];
        coef[0] -= coef[col + 1] * x_mean[col];
    }
    coef[0] += y_mean;
    return coef;
}

Matrix ridge_predict_for_lambdas(
    const Matrix& design,
    const Matrix& target_design,
    const Matrix& response,
    const std::vector<double>& weights,
    const std::vector<double>& lambdas,
    std::vector<Matrix>& fitted_values
) {
    std::vector<Matrix> predictions;
    for (double lambda : lambdas) {
        Matrix fitted(response.rows(), response.cols(), 0.0);
        Matrix prediction(response.rows(), response.cols(), 0.0);

        for (std::size_t par = 0; par < response.cols(); ++par) {
            const std::vector<double> coef = ridge_lm_coefficients(
                design,
                response.col(par),
                weights,
                lambda
            );

            for (std::size_t row = 0; row < response.rows(); ++row) {
                double value = coef[0];
                for (std::size_t col = 1; col < design.cols(); ++col) {
                    value += design(row, col) * coef[col];
                }
                fitted(row, par) = value;
            }

            double target_value = coef[0];
            for (std::size_t col = 1; col < target_design.cols(); ++col) {
                target_value += target_design(0, col) * coef[col];
            }
            for (std::size_t row = 0; row < response.rows(); ++row) {
                prediction(row, par) = target_value;
            }
        }

        fitted_values.push_back(fitted);
        predictions.push_back(prediction);
    }

    return median_matrices(predictions);
}

Matrix run_ridge_like(
    const Matrix& design,
    const Matrix& target_design,
    const Matrix& response,
    const std::vector<double>& weights,
    const std::vector<double>& lambdas,
    bool hcorr,
    Matrix& residuals
) {
    std::vector<Matrix> fitted_values;
    Matrix prediction = ridge_predict_for_lambdas(
        design,
        target_design,
        response,
        weights,
        lambdas,
        fitted_values
    );
    const Matrix fitted = median_matrices(fitted_values);
    residuals = residual_matrix(response, fitted);

    if (hcorr) {
        std::vector<Matrix> sd_fitted;
        const Matrix log_residuals = log_square(residuals);
        const Matrix target_sd_log = ridge_predict_for_lambdas(
            design,
            target_design,
            log_residuals,
            weights,
            lambdas,
            sd_fitted
        );
        const Matrix local_sd_log = median_matrices(sd_fitted);
        Matrix adjusted(response.rows(), response.cols(), 0.0);
        for (std::size_t row = 0; row < adjusted.rows(); ++row) {
            for (std::size_t col = 0; col < adjusted.cols(); ++col) {
                const double target_sd = std::sqrt(
                    std::exp(target_sd_log(row, col))
                );
                const double local_sd = std::sqrt(
                    std::exp(local_sd_log(row, col))
                );
                adjusted(row, col) = prediction(row, col) +
                    target_sd * residuals(row, col) /
                    std::max(local_sd, kTiny);
            }
        }
        residuals = residual_matrix(adjusted, prediction);
        return adjusted;
    }

    Matrix adjusted(response.rows(), response.cols(), 0.0);
    for (std::size_t row = 0; row < adjusted.rows(); ++row) {
        for (std::size_t col = 0; col < adjusted.cols(); ++col) {
            adjusted(row, col) = prediction(row, col) + residuals(row, col);
        }
    }
    return adjusted;
}

double dot_vector(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    double total = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        total += left[i] * right[i];
    }
    return total;
}

double vector_norm(const std::vector<double>& values) {
    return std::sqrt(dot_vector(values, values));
}

std::vector<double> matrix_vector_product(
    const Matrix& matrix,
    const std::vector<double>& vector
) {
    std::vector<double> out(matrix.rows(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            out[row] += matrix(row, col) * vector[col];
        }
    }
    return out;
}

Matrix identity_matrix(std::size_t size) {
    Matrix out(size, size, 0.0);
    for (std::size_t i = 0; i < size; ++i) {
        out(i, i) = 1.0;
    }
    return out;
}

double clipped_sigmoid(double value) {
    if (value < -15.0) {
        return 0.0;
    }
    if (value > 15.0) {
        return 1.0;
    }
    return 1.0 / (1.0 + std::exp(-value));
}

std::size_t nnet_weight_count(
    std::size_t inputs,
    std::size_t hidden,
    std::size_t outputs
) {
    return hidden * (inputs + 1) + outputs * (hidden + 1);
}

std::vector<double> nnet_forward(
    const std::vector<double>& input,
    const std::vector<double>& wts,
    std::size_t hidden,
    std::size_t outputs
) {
    const std::size_t inputs = input.size();
    std::vector<double> hidden_values(hidden, 0.0);
    std::vector<double> out(outputs, 0.0);

    for (std::size_t h = 0; h < hidden; ++h) {
        const std::size_t base = h * (inputs + 1);
        double sum = wts[base];
        for (std::size_t col = 0; col < inputs; ++col) {
            sum += input[col] * wts[base + 1 + col];
        }
        hidden_values[h] = clipped_sigmoid(sum);
    }

    const std::size_t output_offset = hidden * (inputs + 1);
    for (std::size_t out_col = 0; out_col < outputs; ++out_col) {
        const std::size_t base = output_offset + out_col * (hidden + 1);
        double sum = wts[base];
        for (std::size_t h = 0; h < hidden; ++h) {
            sum += hidden_values[h] * wts[base + 1 + h];
        }
        out[out_col] = sum;
    }

    return out;
}

double nnet_value_gradient(
    const Matrix& x,
    const Matrix& response,
    const std::vector<double>& weights,
    const std::vector<double>& wts,
    std::size_t hidden,
    double decay,
    std::vector<double>& gradient
) {
    const std::size_t inputs = x.cols();
    const std::size_t outputs = response.cols();
    gradient.assign(wts.size(), 0.0);

    double objective = 0.0;
    std::vector<double> input(inputs, 0.0);
    std::vector<double> hidden_values(hidden, 0.0);
    std::vector<double> output_values(outputs, 0.0);
    std::vector<double> output_delta(outputs, 0.0);
    std::vector<double> hidden_error(hidden, 0.0);

    const std::size_t output_offset = hidden * (inputs + 1);
    for (std::size_t row = 0; row < x.rows(); ++row) {
        for (std::size_t col = 0; col < inputs; ++col) {
            input[col] = x(row, col);
        }

        for (std::size_t h = 0; h < hidden; ++h) {
            const std::size_t base = h * (inputs + 1);
            double sum = wts[base];
            for (std::size_t col = 0; col < inputs; ++col) {
                sum += input[col] * wts[base + 1 + col];
            }
            hidden_values[h] = clipped_sigmoid(sum);
            hidden_error[h] = 0.0;
        }

        const double row_weight = std::max(weights[row], 0.0);
        for (std::size_t out_col = 0; out_col < outputs; ++out_col) {
            const std::size_t base = output_offset +
                out_col * (hidden + 1);
            double sum = wts[base];
            for (std::size_t h = 0; h < hidden; ++h) {
                sum += hidden_values[h] * wts[base + 1 + h];
            }
            output_values[out_col] = sum;

            const double diff = sum - response(row, out_col);
            objective += row_weight * diff * diff;
            output_delta[out_col] = 2.0 * diff;

            gradient[base] += row_weight * output_delta[out_col];
            for (std::size_t h = 0; h < hidden; ++h) {
                gradient[base + 1 + h] += row_weight *
                    output_delta[out_col] * hidden_values[h];
                hidden_error[h] += output_delta[out_col] *
                    wts[base + 1 + h];
            }
        }

        for (std::size_t h = 0; h < hidden; ++h) {
            const double slope = hidden_values[h] * (1.0 - hidden_values[h]);
            const double delta = hidden_error[h] * slope;
            const std::size_t base = h * (inputs + 1);
            gradient[base] += row_weight * delta;
            for (std::size_t col = 0; col < inputs; ++col) {
                gradient[base + 1 + col] += row_weight * delta * input[col];
            }
        }
    }

    for (std::size_t i = 0; i < wts.size(); ++i) {
        objective += decay * wts[i] * wts[i];
        gradient[i] += 2.0 * decay * wts[i];
    }

    return objective;
}

std::vector<double> bfgs_train_nnet(
    const Matrix& x,
    const Matrix& response,
    const std::vector<double>& weights,
    std::size_t hidden,
    double decay,
    int maxit,
    std::mt19937& rng
) {
    const std::size_t nwts = nnet_weight_count(
        x.cols(),
        hidden,
        response.cols()
    );
    std::uniform_real_distribution<double> uniform(-0.7, 0.7);
    std::vector<double> current(nwts, 0.0);
    for (double& value : current) {
        value = uniform(rng);
    }

    Matrix inverse_hessian = identity_matrix(nwts);
    std::vector<double> gradient;
    double value = nnet_value_gradient(
        x,
        response,
        weights,
        current,
        hidden,
        decay,
        gradient
    );

    for (int iter = 0; iter < std::max(1, maxit); ++iter) {
        if (!std::isfinite(value) || vector_norm(gradient) < 1e-6) {
            break;
        }

        std::vector<double> direction = matrix_vector_product(
            inverse_hessian,
            gradient
        );
        for (double& component : direction) {
            component = -component;
        }

        double directional_derivative = dot_vector(gradient, direction);
        if (directional_derivative >= 0.0 ||
            !std::isfinite(directional_derivative)) {
            inverse_hessian = identity_matrix(nwts);
            direction = gradient;
            for (double& component : direction) {
                component = -component;
            }
            directional_derivative = dot_vector(gradient, direction);
        }

        double step = 1.0;
        std::vector<double> candidate(nwts, 0.0);
        std::vector<double> candidate_gradient;
        double candidate_value = std::numeric_limits<double>::infinity();

        for (int line = 0; line < 30; ++line) {
            for (std::size_t i = 0; i < nwts; ++i) {
                candidate[i] = current[i] + step * direction[i];
            }
            candidate_value = nnet_value_gradient(
                x,
                response,
                weights,
                candidate,
                hidden,
                decay,
                candidate_gradient
            );

            const double armijo = value + 1e-4 * step *
                directional_derivative;
            if (std::isfinite(candidate_value) && candidate_value <= armijo) {
                break;
            }
            step *= 0.5;
        }

        if (!std::isfinite(candidate_value) || step < 1e-9) {
            break;
        }

        std::vector<double> s(nwts, 0.0);
        std::vector<double> y(nwts, 0.0);
        for (std::size_t i = 0; i < nwts; ++i) {
            s[i] = candidate[i] - current[i];
            y[i] = candidate_gradient[i] - gradient[i];
        }

        const double ys = dot_vector(y, s);
        if (ys > 1e-12 && std::isfinite(ys)) {
            const std::vector<double> hy = matrix_vector_product(
                inverse_hessian,
                y
            );
            const double yhy = dot_vector(y, hy);
            const double coeff = (ys + yhy) / (ys * ys);

            for (std::size_t row = 0; row < nwts; ++row) {
                for (std::size_t col = 0; col < nwts; ++col) {
                    inverse_hessian(row, col) += coeff * s[row] * s[col] -
                        (s[row] * hy[col] + hy[row] * s[col]) / ys;
                }
            }
        } else {
            inverse_hessian = identity_matrix(nwts);
        }

        const double old_value = value;
        current = candidate;
        gradient = candidate_gradient;
        value = candidate_value;

        const double improvement = std::fabs(old_value - value);
        const double scale = std::max(1.0, std::fabs(old_value));
        if (improvement <= 1e-8 * scale) {
            break;
        }
    }

    return current;
}

Matrix nnet_predict_matrix(
    const Matrix& x,
    const std::vector<double>& wts,
    std::size_t hidden,
    std::size_t outputs
) {
    Matrix out(x.rows(), outputs, 0.0);
    for (std::size_t row = 0; row < x.rows(); ++row) {
        const std::vector<double> pred = nnet_forward(
            x.row(row),
            wts,
            hidden,
            outputs
        );
        out.set_row(row, pred);
    }
    return out;
}

Matrix nnet_repeat_target_prediction(
    const std::vector<double>& target,
    const std::vector<double>& wts,
    std::size_t rows,
    std::size_t hidden,
    std::size_t outputs
) {
    const std::vector<double> pred = nnet_forward(
        target,
        wts,
        hidden,
        outputs
    );
    Matrix out(rows, outputs, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        out.set_row(row, pred);
    }
    return out;
}

struct NNetEnsemble {
    Matrix fitted;
    Matrix prediction;
};

NNetEnsemble train_nnet_ensemble(
    const Matrix& x,
    const std::vector<double>& target,
    const Matrix& response,
    const std::vector<double>& weights,
    const std::vector<double>& lambdas,
    std::size_t hidden,
    int maxit,
    std::mt19937& rng
) {
    std::vector<Matrix> fitted_values;
    std::vector<Matrix> predictions;

    for (double lambda : lambdas) {
        const std::vector<double> wts = bfgs_train_nnet(
            x,
            response,
            weights,
            hidden,
            lambda,
            maxit,
            rng
        );
        fitted_values.push_back(nnet_predict_matrix(
            x,
            wts,
            hidden,
            response.cols()
        ));
        predictions.push_back(nnet_repeat_target_prediction(
            target,
            wts,
            response.rows(),
            hidden,
            response.cols()
        ));
    }

    return NNetEnsemble{
        median_matrices(fitted_values),
        median_matrices(predictions)
    };
}

Matrix run_neural_like(
    const Matrix& x,
    const std::vector<double>& target,
    const Matrix& response,
    const std::vector<double>& weights,
    const AbcOptions& options,
    Matrix& residuals,
    std::vector<double>& lambda_used
) {
    const int net_count = std::max(1, options.numnet);
    const std::size_t hidden = static_cast<std::size_t>(
        std::max(1, options.sizenet)
    );

    std::mt19937 rng(options.seed);
    std::uniform_int_distribution<std::size_t> lambda_index(
        0,
        options.lambda.size() - 1
    );

    lambda_used.clear();
    lambda_used.reserve(static_cast<std::size_t>(net_count));
    for (int net = 0; net < net_count; ++net) {
        lambda_used.push_back(options.lambda[lambda_index(rng)]);
    }

    const NNetEnsemble primary = train_nnet_ensemble(
        x,
        target,
        response,
        weights,
        lambda_used,
        hidden,
        options.maxit,
        rng
    );

    Matrix prediction = primary.prediction;
    residuals = residual_matrix(response, primary.fitted);

    if (options.hcorr) {
        const Matrix log_residuals = log_square(residuals);
        const NNetEnsemble scale_model = train_nnet_ensemble(
            x,
            target,
            log_residuals,
            weights,
            lambda_used,
            hidden,
            options.maxit,
            rng
        );

        Matrix adjusted(response.rows(), response.cols(), 0.0);
        for (std::size_t row = 0; row < adjusted.rows(); ++row) {
            for (std::size_t col = 0; col < adjusted.cols(); ++col) {
                const double target_sd = std::sqrt(std::exp(
                    scale_model.prediction(row, col)
                ));
                const double local_sd = std::sqrt(std::exp(
                    scale_model.fitted(row, col)
                ));
                adjusted(row, col) = prediction(row, col) +
                    target_sd * residuals(row, col) /
                    std::max(local_sd, kTiny);
            }
        }
        residuals = residual_matrix(adjusted, prediction);
        return adjusted;
    }

    Matrix adjusted(response.rows(), response.cols(), 0.0);
    for (std::size_t row = 0; row < adjusted.rows(); ++row) {
        for (std::size_t col = 0; col < adjusted.cols(); ++col) {
            adjusted(row, col) = prediction(row, col) + residuals(row, col);
        }
    }
    return adjusted;
}

bool has_variance(const Matrix& matrix, std::size_t col) {
    if (matrix.rows() < 2) {
        return false;
    }

    const double first = matrix(0, col);
    for (std::size_t row = 1; row < matrix.rows(); ++row) {
        if (std::fabs(matrix(row, col) - first) > 1e-14) {
            return true;
        }
    }
    return false;
}

std::vector<double> flatten_matrix(const Matrix& matrix) {
    return matrix.data();
}

Matrix reshape_sumstat_for_param(
    const Matrix& sumstat,
    std::size_t simulation_count
) {
    if (sumstat.rows() == simulation_count) {
        return sumstat;
    }

    if (simulation_count == 0 ||
        sumstat.data().size() % simulation_count != 0) {
        throw std::invalid_argument("param and sumstat must share rows.");
    }

    const std::size_t cols = sumstat.data().size() / simulation_count;
    return from_row_major(sumstat.data(), simulation_count, cols);
}

}  // namespace

AbcResult abc(
    const std::vector<double>& raw_target,
    const Matrix& raw_param,
    const Matrix& raw_sumstat,
    const AbcOptions& raw_options
) {
    if (raw_param.rows() != raw_sumstat.rows()) {
        throw std::invalid_argument("param and sumstat must share rows.");
    }
    if (raw_target.size() != raw_sumstat.cols()) {
        throw std::invalid_argument("target and sumstat must share columns.");
    }
    if (!raw_options.subset.empty() &&
        raw_options.subset.size() != raw_sumstat.rows()) {
        throw std::invalid_argument("subset has wrong length.");
    }
    if (raw_options.lambda.empty()) {
        throw std::invalid_argument("lambda must contain at least one value.");
    }

    AbcOptions options = raw_options;

    /* =========================
     * Summary Reduction
     * ========================= */

    ReducedSummary reduced = reduce_summary_statistics(
        raw_param,
        raw_sumstat,
        raw_target,
        options.reduction
    );

    Matrix param = raw_param;
    Matrix sumstat = reduced.sumstat;
    std::vector<double> target = reduced.target;
    const std::vector<Transform> transforms = normalized_transforms(
        options,
        param.cols()
    );

    /* =========================
     * Input Filtering
     * ========================= */

    const std::vector<bool> good_rows = build_good_rows(
        param,
        sumstat,
        options.subset
    );

    std::size_t good_count = 0;
    for (bool keep : good_rows) {
        if (keep) {
            ++good_count;
        }
    }
    if (good_count == 0) {
        throw std::invalid_argument("No valid simulation row is available.");
    }

    Matrix scaled_sumstat = scale_sumstat(sumstat, good_rows, target);
    std::vector<double> distances = compute_distances(
        scaled_sumstat,
        target,
        good_rows
    );

    std::vector<bool> region = select_region(
        distances,
        good_rows,
        options.tol,
        options.kernel
    );
    const double cutoff = distance_cutoff(distances, good_rows, options.tol);

    std::vector<std::size_t> accepted_rows;
    for (std::size_t row = 0; row < region.size(); ++row) {
        if (region[row]) {
            accepted_rows.push_back(row);
        }
    }
    if (accepted_rows.empty()) {
        throw std::runtime_error("No simulation row was accepted.");
    }

    /* =========================
     * Parameter Transformation
     * ========================= */

    Matrix transformed_param = transform_parameters(
        param,
        transforms,
        options.logit_bounds
    );

    Matrix accepted_sumstat = accepted_matrix(sumstat, accepted_rows);
    Matrix unadjusted = accepted_matrix(transformed_param, accepted_rows);

    bool any_var = false;
    for (std::size_t col = 0; col < accepted_sumstat.cols(); ++col) {
        any_var = any_var || has_variance(accepted_sumstat, col);
    }
    if (!any_var && options.method != Method::Rejection) {
        throw std::runtime_error(
            "Zero variance in accepted summary statistics."
        );
    }

    AbcResult result;
    result.unadj_values = unadjusted;
    result.accepted_sumstats = accepted_sumstat;
    result.distances = distances;
    result.accepted_indices = accepted_rows;
    result.region = region;
    result.na_action = good_rows;
    result.transformations = transforms;
    result.logit_bounds = options.logit_bounds;
    result.options = options;
    result.method = options.method;
    result.kernel = options.kernel;
    result.hcorr = options.hcorr;
    result.numparam = static_cast<int>(param.cols());
    result.numstat = static_cast<int>(sumstat.cols());
    result.reduction = reduced.info;

    if (options.method == Method::Rejection) {
        result.weights = Matrix(accepted_rows.size(), 1, 1.0);
        result.diagnostics.aic = result.aic;
        result.diagnostics.bic = result.bic;
        result.diagnostics.lambda = result.lambda;
        back_transform(result.unadj_values, transforms, options.logit_bounds);
        return result;
    }

    /* =========================
     * Regression Adjustment
     * ========================= */

    const std::vector<double> weights_vec = build_weights(
        distances,
        accepted_rows,
        cutoff,
        options.kernel
    );

    result.weights = Matrix(weights_vec.size(), 1, 0.0);
    for (std::size_t i = 0; i < weights_vec.size(); ++i) {
        result.weights(i, 0) = weights_vec[i];
    }

    const Matrix x = accepted_matrix(scaled_sumstat, accepted_rows);
    const Matrix design = add_intercept(x);
    const Matrix target_design = make_target_design(target);
    Matrix response = unadjusted;
    Matrix residuals;

    if (options.method == Method::LocLinear) {
        result.adj_values = run_loclinear(
            design,
            target_design,
            response,
            weights_vec,
            options.hcorr,
            result.aic,
            result.bic,
            residuals
        );
    }

    if (options.method == Method::Ridge) {
        const std::vector<double> scales = param_mads(
            transformed_param,
            good_rows
        );
        response = divide_columns(response, scales);
        result.adj_values = run_ridge_like(
            design,
            target_design,
            response,
            weights_vec,
            options.lambda,
            options.hcorr,
            residuals
        );
        multiply_columns(result.adj_values, scales);
    }

    if (options.method == Method::NeuralNet) {
        const std::vector<double> scales = param_mads(
            transformed_param,
            good_rows
        );
        response = divide_columns(response, scales);
        result.adj_values = run_neural_like(
            x,
            target,
            response,
            weights_vec,
            options,
            residuals,
            result.lambda
        );
        multiply_columns(result.adj_values, scales);
    }

    result.residuals = residuals;
    result.diagnostics.aic = result.aic;
    result.diagnostics.bic = result.bic;
    result.diagnostics.lambda = result.lambda;

    back_transform(result.unadj_values, transforms, options.logit_bounds);
    back_transform(result.adj_values, transforms, options.logit_bounds);
    return result;
}

AbcResult abc(
    const Matrix& raw_target,
    const Matrix& raw_param,
    const Matrix& raw_sumstat,
    const AbcOptions& raw_options
) {
    const Matrix prepared_sumstat = reshape_sumstat_for_param(
        raw_sumstat,
        raw_param.rows()
    );

    return abc(
        flatten_matrix(raw_target),
        raw_param,
        prepared_sumstat,
        raw_options
    );
}

}  // namespace abcpp
