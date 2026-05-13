#include <Rcpp.h>
#include <algorithm>
#include "../../Cpp/include/progress_bar.hpp"

#define UI_PROGRESS_IMPL "../../Cpp/src/progress_bar.cpp"
#include UI_PROGRESS_IMPL

// [[Rcpp::export(name = "progress_start")]]
void r_progress_start(
    double total,
    std::string title = "Progress",
    int refresh_ms = 100,
    std::string mode = "auto",
    double line_interval_sec = 2.0,
    double line_interval_pct = 5.0
) {
    ui::ProgressOptions options;
    options.mode = mode;
    options.refresh_ms = refresh_ms;
    options.line_interval_sec = line_interval_sec;
    options.line_interval_pct = line_interval_pct;
    ui::progress_start(
        static_cast<std::size_t>(std::max(0.0, total)),
        title,
        refresh_ms,
        options
    );
}

// [[Rcpp::export(name = "progress_set")]]
void r_progress_set(double current) {
    ui::progress_set(static_cast<std::size_t>(std::max(0.0, current)));
}

// [[Rcpp::export(name = "progress_advance")]]
void r_progress_advance(double step = 1.0) {
    ui::progress_advance(static_cast<std::size_t>(std::max(0.0, step)));
}

// [[Rcpp::export(name = "progress_finish")]]
void r_progress_finish() {
    ui::progress_finish();
}

// [[Rcpp::export(name = "progress_snapshot")]]
Rcpp::List r_progress_snapshot() {
    ui::ProgressSnapshot snapshot = ui::progress_last_snapshot();
    return Rcpp::List::create(
        Rcpp::Named("title") = snapshot.title,
        Rcpp::Named("requested_mode") = snapshot.requested_mode,
        Rcpp::Named("resolved_mode") = snapshot.resolved_mode,
        Rcpp::Named("elapsed_sec") = snapshot.elapsed_sec,
        Rcpp::Named("percent") = snapshot.percent,
        Rcpp::Named("eta_sec") = snapshot.eta_sec,
        Rcpp::Named("speed") = snapshot.speed,
        Rcpp::Named("started") = snapshot.started,
        Rcpp::Named("running") = snapshot.running,
        Rcpp::Named("finished") = snapshot.finished,
        Rcpp::Named("total_iterations") =
            static_cast<double>(snapshot.total),
        Rcpp::Named("completed_iterations") =
            static_cast<double>(snapshot.completed)
    );
}
