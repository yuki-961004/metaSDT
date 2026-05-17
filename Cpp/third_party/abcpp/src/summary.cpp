#include "abcpp/summary.hpp"

#include "abcpp/statistics.hpp"

namespace abcpp {

SummaryResult summary(
    const AbcResult& result,
    bool unadjusted,
    double interval
) {
    const Matrix& values = (unadjusted ||
        result.method == Method::Rejection ||
        result.adj_values.empty())
        ? result.unadj_values
        : result.adj_values;

    SummaryResult out;
    out.interval = interval;
    out.unadjusted = unadjusted;
    out.columns.resize(values.cols());

    const double lower = (1.0 - interval) / 2.0;
    const double upper = 1.0 - lower;

    std::vector<double> weights;
    if (!result.weights.empty() && result.weights.rows() == values.rows()) {
        for (std::size_t row = 0; row < result.weights.rows(); ++row) {
            weights.push_back(result.weights(row, 0));
        }
    }

    for (std::size_t col = 0; col < values.cols(); ++col) {
        std::vector<double> column = values.col(col);
        SummaryColumn column_summary;
        column_summary.min = quantile_type7(column, 0.0);
        column_summary.q_lower = quantile_type7(column, lower);
        column_summary.median = quantile_type7(column, 0.5);
        column_summary.mean = mean(column);
        column_summary.mode = kernel_mode(column, weights);
        column_summary.q_upper = quantile_type7(column, upper);
        column_summary.max = quantile_type7(column, 1.0);
        column_summary.sd = sample_sd(column);
        out.columns[col] = column_summary;
    }

    return out;
}

SummaryResult summarize(
    const AbcResult& result,
    bool unadjusted,
    double interval
) {
    return summary(result, unadjusted, interval);
}

}  // namespace abcpp
