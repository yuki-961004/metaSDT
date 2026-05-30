#pragma once

#include <cstddef>
#include <string>

namespace ui {

struct ProgressOptions {
    std::string mode = "auto"; // auto | dynamic | line | silent
    int refresh_ms = 100;
    double line_interval_sec = 2.0;
    double line_interval_pct = 5.0;
};

struct ProgressSnapshot {
    std::string title;
    std::size_t total = 0;
    std::size_t completed = 0;
    double elapsed_sec = 0.0;
    double percent = 0.0;
    double eta_sec = 0.0;
    double speed = 0.0;
    bool started = false;
    bool running = false;
    bool finished = false;
    std::string requested_mode = "auto";
    std::string resolved_mode = "silent";
};

void progress_start(
    std::size_t total,
    const std::string& title = "Progress",
    int refresh_ms = 100,
    const ProgressOptions& options = ProgressOptions{}
);
void progress_set(std::size_t current);
void progress_advance(std::size_t step = 1);
void progress_finish();
ProgressSnapshot progress_last_snapshot();

} // namespace ui
