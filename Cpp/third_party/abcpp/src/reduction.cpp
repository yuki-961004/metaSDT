#include "abcpp/reduction.hpp"

#include "abcpp/linear_algebra.hpp"
#include "abcpp/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace abcpp {

namespace {

Matrix identity(std::size_t size) {
    Matrix out(size, size, 0.0);
    for (std::size_t i = 0; i < size; ++i) {
        out(i, i) = 1.0;
    }
    return out;
}

double dot(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    double total = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        total += left[i] * right[i];
    }
    return total;
}

double norm(const std::vector<double>& values) {
    return std::sqrt(dot(values, values));
}

std::vector<double> center_columns(Matrix& matrix) {
    std::vector<double> centers(matrix.cols(), 0.0);
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
        centers[col] = mean(matrix.col(col));
        for (std::size_t row = 0; row < matrix.rows(); ++row) {
            matrix(row, col) -= centers[col];
        }
    }
    return centers;
}

Matrix covariance_matrix(const Matrix& centered) {
    Matrix out(centered.cols(), centered.cols(), 0.0);
    const double denom = centered.rows() > 1
        ? static_cast<double>(centered.rows() - 1)
        : 1.0;

    for (std::size_t i = 0; i < centered.cols(); ++i) {
        for (std::size_t j = i; j < centered.cols(); ++j) {
            double total = 0.0;
            for (std::size_t row = 0; row < centered.rows(); ++row) {
                total += centered(row, i) * centered(row, j);
            }
            out(i, j) = total / denom;
            out(j, i) = out(i, j);
        }
    }
    return out;
}

struct EigenResult {
    std::vector<double> values;
    Matrix vectors;
};

EigenResult jacobi_eigen(Matrix matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Eigen decomposition requires a square matrix.");
    }

    const std::size_t n = matrix.rows();
    Matrix vectors = identity(n);

    for (std::size_t iter = 0; iter < 100 * n * n + 1; ++iter) {
        std::size_t p = 0;
        std::size_t q = 1;
        double max_abs = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const double candidate = std::fabs(matrix(i, j));
                if (candidate > max_abs) {
                    max_abs = candidate;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_abs < 1e-12 || n < 2) {
            break;
        }

        const double app = matrix(p, p);
        const double aqq = matrix(q, q);
        const double apq = matrix(p, q);
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(angle);
        const double s = std::sin(angle);

        for (std::size_t k = 0; k < n; ++k) {
            const double mkp = matrix(k, p);
            const double mkq = matrix(k, q);
            matrix(k, p) = c * mkp - s * mkq;
            matrix(k, q) = s * mkp + c * mkq;
        }

        for (std::size_t k = 0; k < n; ++k) {
            const double mpk = matrix(p, k);
            const double mqk = matrix(q, k);
            matrix(p, k) = c * mpk - s * mqk;
            matrix(q, k) = s * mpk + c * mqk;
        }

        for (std::size_t k = 0; k < n; ++k) {
            const double vkp = vectors(k, p);
            const double vkq = vectors(k, q);
            vectors(k, p) = c * vkp - s * vkq;
            vectors(k, q) = s * vkp + c * vkq;
        }
    }

    std::vector<double> values(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = matrix(i, i);
    }

    return EigenResult{values, vectors};
}

Matrix select_sorted_components(
    const EigenResult& eigen,
    std::size_t ncomp
) {
    std::vector<std::size_t> order(eigen.values.size(), 0);
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    std::sort(
        order.begin(),
        order.end(),
        [&](std::size_t left, std::size_t right) {
            return eigen.values[left] > eigen.values[right];
        }
    );

    Matrix components(eigen.vectors.rows(), ncomp, 0.0);
    for (std::size_t comp = 0; comp < ncomp; ++comp) {
        for (std::size_t row = 0; row < eigen.vectors.rows(); ++row) {
            components(row, comp) = eigen.vectors(row, order[comp]);
        }
    }
    return components;
}

ReducedSummary reduce_pca(
    const Matrix& sumstat,
    const std::vector<double>& target,
    std::size_t requested_ncomp
) {
    Matrix centered = sumstat;
    const std::vector<double> centers = center_columns(centered);
    const std::size_t ncomp = std::max<std::size_t>(
        1,
        std::min(requested_ncomp, sumstat.cols())
    );

    const EigenResult eigen = jacobi_eigen(covariance_matrix(centered));
    const Matrix components = select_sorted_components(eigen, ncomp);
    const Matrix scores = multiply(centered, components);

    Matrix target_centered(1, target.size(), 0.0);
    for (std::size_t col = 0; col < target.size(); ++col) {
        target_centered(0, col) = target[col] - centers[col];
    }

    const Matrix target_scores = multiply(target_centered, components);
    std::vector<double> reduced_target(ncomp, 0.0);
    for (std::size_t col = 0; col < ncomp; ++col) {
        reduced_target[col] = target_scores(0, col);
    }

    ReductionInfo info;
    info.method = ReductionMethod::PCA;
    info.ncomp = ncomp;
    info.rotation = components;
    info.center = centers;

    return ReducedSummary{scores, reduced_target, info};
}

ReducedSummary reduce_pls(
    const Matrix& param,
    const Matrix& sumstat,
    const std::vector<double>& target,
    std::size_t requested_ncomp
) {
    Matrix x_original = sumstat;
    Matrix x_residual = sumstat;
    Matrix y_residual = param;
    const std::vector<double> x_centers = center_columns(x_residual);
    center_columns(y_residual);

    for (std::size_t row = 0; row < x_original.rows(); ++row) {
        for (std::size_t col = 0; col < x_original.cols(); ++col) {
            x_original(row, col) -= x_centers[col];
        }
    }

    const std::size_t max_comp = std::min(sumstat.cols(), sumstat.rows());
    const std::size_t ncomp = std::max<std::size_t>(
        1,
        std::min(requested_ncomp, max_comp)
    );

    Matrix weights(sumstat.cols(), ncomp, 0.0);
    Matrix loadings(sumstat.cols(), ncomp, 0.0);

    for (std::size_t comp = 0; comp < ncomp; ++comp) {
        std::size_t y_col = 0;
        double best_norm = -1.0;
        for (std::size_t col = 0; col < y_residual.cols(); ++col) {
            const double candidate = dot(
                y_residual.col(col),
                y_residual.col(col)
            );
            if (candidate > best_norm) {
                best_norm = candidate;
                y_col = col;
            }
        }

        std::vector<double> u = y_residual.col(y_col);
        std::vector<double> w(sumstat.cols(), 0.0);
        std::vector<double> t(sumstat.rows(), 0.0);
        std::vector<double> q(param.cols(), 0.0);

        for (std::size_t iter = 0; iter < 100; ++iter) {
            std::fill(w.begin(), w.end(), 0.0);
            for (std::size_t col = 0; col < x_residual.cols(); ++col) {
                for (std::size_t row = 0; row < x_residual.rows(); ++row) {
                    w[col] += x_residual(row, col) * u[row];
                }
            }

            const double w_norm = norm(w);
            if (w_norm <= 1e-12) {
                break;
            }
            for (double& value : w) {
                value /= w_norm;
            }

            std::fill(t.begin(), t.end(), 0.0);
            for (std::size_t row = 0; row < x_residual.rows(); ++row) {
                for (std::size_t col = 0; col < x_residual.cols(); ++col) {
                    t[row] += x_residual(row, col) * w[col];
                }
            }

            const double denom_t = dot(t, t);
            if (denom_t <= 1e-12) {
                break;
            }

            std::fill(q.begin(), q.end(), 0.0);
            for (std::size_t col = 0; col < y_residual.cols(); ++col) {
                for (std::size_t row = 0; row < y_residual.rows(); ++row) {
                    q[col] += y_residual(row, col) * t[row] / denom_t;
                }
            }

            const double denom_q = dot(q, q);
            if (denom_q <= 1e-12) {
                break;
            }

            std::vector<double> next_u(y_residual.rows(), 0.0);
            for (std::size_t row = 0; row < y_residual.rows(); ++row) {
                for (std::size_t col = 0; col < y_residual.cols(); ++col) {
                    next_u[row] += y_residual(row, col) * q[col] / denom_q;
                }
            }

            double delta = 0.0;
            for (std::size_t row = 0; row < u.size(); ++row) {
                const double diff = next_u[row] - u[row];
                delta += diff * diff;
            }
            u = next_u;
            if (delta < 1e-16) {
                break;
            }
        }

        const double denom_t = dot(t, t);
        if (denom_t <= 1e-12) {
            break;
        }

        std::vector<double> p(sumstat.cols(), 0.0);
        for (std::size_t col = 0; col < x_residual.cols(); ++col) {
            for (std::size_t row = 0; row < x_residual.rows(); ++row) {
                p[col] += x_residual(row, col) * t[row] / denom_t;
            }
        }

        for (std::size_t row = 0; row < x_residual.rows(); ++row) {
            for (std::size_t col = 0; col < x_residual.cols(); ++col) {
                x_residual(row, col) -= t[row] * p[col];
            }
            for (std::size_t col = 0; col < y_residual.cols(); ++col) {
                y_residual(row, col) -= t[row] * q[col];
            }
        }

        for (std::size_t row = 0; row < sumstat.cols(); ++row) {
            weights(row, comp) = w[row];
            loadings(row, comp) = p[row];
        }
    }

    const Matrix projection = multiply(
        weights,
        inverse(multiply(transpose(loadings), weights))
    );
    const Matrix scores = multiply(x_original, projection);

    Matrix target_centered(1, target.size(), 0.0);
    for (std::size_t col = 0; col < target.size(); ++col) {
        target_centered(0, col) = target[col] - x_centers[col];
    }

    const Matrix target_scores = multiply(target_centered, projection);
    std::vector<double> reduced_target(ncomp, 0.0);
    for (std::size_t col = 0; col < ncomp; ++col) {
        reduced_target[col] = target_scores(0, col);
    }

    ReductionInfo info;
    info.method = ReductionMethod::PLS;
    info.ncomp = ncomp;
    info.rotation = projection;
    info.center = x_centers;

    return ReducedSummary{scores, reduced_target, info};
}

}  // namespace

ReducedSummary reduce_summary_statistics(
    const Matrix& param,
    const Matrix& sumstat,
    const std::vector<double>& target,
    const ReductionOptions& options
) {
    if (options.method == ReductionMethod::None) {
        ReductionInfo info;
        info.method = ReductionMethod::None;
        info.ncomp = sumstat.cols();
        return ReducedSummary{sumstat, target, info};
    }

    if (target.size() != sumstat.cols()) {
        throw std::invalid_argument("Target and summary statistics mismatch.");
    }

    const std::size_t ncomp = options.ncomp == 0
        ? std::min(sumstat.cols(), param.cols())
        : options.ncomp;

    if (options.method == ReductionMethod::PCA) {
        return reduce_pca(sumstat, target, ncomp);
    }
    if (options.method == ReductionMethod::PLS) {
        return reduce_pls(param, sumstat, target, ncomp);
    }

    throw std::invalid_argument("Unsupported summary reduction method.");
}

}  // namespace abcpp
