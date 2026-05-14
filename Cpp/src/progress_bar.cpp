#include "../include/progress_bar.hpp"

#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#if defined(_WIN32)
#include <io.h>
#define UI_ISATTY _isatty
#define UI_FILENO _fileno
#else
#include <unistd.h>
#define UI_ISATTY isatty
#define UI_FILENO fileno
#endif

namespace ui {
namespace {

std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool env_true(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::string(v).size() > 0;
}

class ProgressCore {
public:
    void start(
        std::size_t total, const std::string& title, int refresh_ms, 
        const ProgressOptions& options
    ) {
        std::lock_guard<std::mutex> lock(mu_);
        total_ = total;
        current_ = 0;
        title_ = title.empty() ? "Progress" : title;
        opts_ = options;
        if (refresh_ms > 0) opts_.refresh_ms = refresh_ms;
        resolved_mode_ = resolve_mode(opts_.mode);
        running_ = true;
        finished_ = false;
        started_at_ = std::chrono::steady_clock::now();
        last_emit_at_ = started_at_;
        last_emit_pct_ = -1.0;
    }

    void set(std::size_t current) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            current_ = current;
            if (total_ > 0 && current_ > total_) current_ = total_;
        }
        maybe_render(false);
    }

    void advance(std::size_t step) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            current_ += step;
            if (total_ > 0 && current_ > total_) current_ = total_;
        }
        maybe_render(false);
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!running_) return;
            finished_ = true;
            running_ = false;
        }
        maybe_render(true);
        if (resolved_mode_ == "dynamic") {
            std::lock_guard<std::mutex> lk(out_mu_);
            std::cout << "\n";
            std::cout.flush();
        }
    }

    ProgressSnapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        ProgressSnapshot s;
        s.title = title_;
        s.total = total_;
        s.completed = current_;
        const auto now = std::chrono::steady_clock::now();
        s.elapsed_sec = std::chrono::duration<double>(
            now - started_at_
        ).count();
        s.percent = (total_ > 0) ? 
            (100.0 * static_cast<double>(current_) / 
             static_cast<double>(total_)) : 0.0;
        s.speed = (s.elapsed_sec > 1e-9) ? 
            (static_cast<double>(current_) / s.elapsed_sec) : 0.0;
        s.eta_sec = (s.speed > 1e-9 && total_ > current_) ? 
            (static_cast<double>(total_ - current_) / s.speed) : 0.0;
        s.started = started_at_.time_since_epoch().count() != 0;
        s.running = running_;
        s.finished = finished_;
        s.requested_mode = opts_.mode;
        s.resolved_mode = resolved_mode_;
        return s;
    }

private:
    std::string resolve_mode(const std::string& requested) {
        std::string m = to_lower(requested);
        if (m == "dynamic" || m == "line" || m == "silent") return m;

        bool is_tty = UI_ISATTY(UI_FILENO(stdout)) != 0;
        bool in_vscode_like = env_true("VSCODE_PID") || 
            env_true("TERM_PROGRAM") || env_true("POSITRON_VERSION") || 
            env_true("POSITRON_SESSION_ID");
        bool in_jupyter = env_true("JPY_PARENT_PID") || 
            env_true("IPYKERNEL_PARENT_PID") || env_true("JUPYTERHUB_USER");
        bool in_quarto = env_true("QUARTO_PROJECT_ROOT") || 
            env_true("QUARTO_PROFILE") || env_true("QUARTO_RENDER");

        if (in_jupyter || in_quarto || in_vscode_like || !is_tty) return "line";
        return "dynamic";
    }

    void maybe_render(bool force) {
        ProgressSnapshot s = snapshot();
        if (s.resolved_mode == "silent") return;

        auto now = std::chrono::steady_clock::now();
        bool emit = force;

        if (!emit) {
            const double sec_step = opts_.line_interval_sec > 0.0 ? 
                                    opts_.line_interval_sec : 2.0;
            const double pct_step = opts_.line_interval_pct > 0.0 ? 
                                    opts_.line_interval_pct : 5.0;
            const double elapsed_since_emit = 
                std::chrono::duration<double>(now - last_emit_at_).count();
            if (s.resolved_mode == "dynamic") {
                emit = elapsed_since_emit >= 
                    (static_cast<double>(
                        opts_.refresh_ms > 0 ? opts_.refresh_ms : 100
                    ) / 1000.0);
            } else {
                emit = (elapsed_since_emit >= sec_step) || 
                       (s.percent >= last_emit_pct_ + pct_step);
            }
        }

        if (!emit) return;

        std::ostringstream oss;
        if (s.resolved_mode == "dynamic") {
            const int bar_width = 24;
            const int filled = (s.total > 0)
                ? static_cast<int>(std::round(
                    (static_cast<double>(s.completed) / 
                     static_cast<double>(s.total)) * bar_width
                  ))
                : 0;
            oss << "\r" << s.title << " [";
            for (int i = 0; i < bar_width; ++i) oss << (i < filled ? '=' : ' ');
            oss << "] "
                << std::fixed << std::setprecision(1) << s.percent << "% "
                << s.completed << "/" << s.total
                << " | elapsed " << std::setprecision(1) << s.elapsed_sec << "s"
                << " | eta " << std::setprecision(1) << s.eta_sec << "s"
                << " | speed " << std::setprecision(1) << s.speed << "/s";
            if (force || s.finished) oss << " | done";
            std::lock_guard<std::mutex> lk(out_mu_);
            std::cout << oss.str();
            std::cout.flush();
        } else {
            oss << s.title << ": "
                << std::fixed << std::setprecision(1) << s.percent << "%"
                << " | " << s.completed << "/" << s.total
                << " | elapsed " << std::setprecision(1) << s.elapsed_sec << "s"
                << " | eta " << std::setprecision(1) << s.eta_sec << "s"
                << " | speed " << std::setprecision(1) << s.speed << "/s";
            if (force || s.finished) oss << " | done";
            std::lock_guard<std::mutex> lk(out_mu_);
            std::cout << oss.str() << "\n";
            std::cout.flush();
        }

        std::lock_guard<std::mutex> lock(mu_);
        last_emit_at_ = now;
        last_emit_pct_ = s.percent;
    }

private:
    mutable std::mutex mu_;
    std::mutex out_mu_;
    std::size_t total_ = 0;
    std::size_t current_ = 0;
    std::string title_ = "Progress";
    ProgressOptions opts_{};
    std::string resolved_mode_ = "silent";
    bool running_ = false;
    bool finished_ = false;
    std::chrono::steady_clock::time_point started_at_{};
    std::chrono::steady_clock::time_point last_emit_at_{};
    double last_emit_pct_ = -1.0;
};

ProgressCore& core() {
    static ProgressCore instance;
    return instance;
}

} // namespace

void progress_start(
    std::size_t total, const std::string& title, int refresh_ms, 
    const ProgressOptions& options
) {
    core().start(total, title, refresh_ms, options);
}

void progress_set(std::size_t current) { core().set(current); }
void progress_advance(std::size_t step) { core().advance(step); }
void progress_finish() { core().finish(); }
ProgressSnapshot progress_last_snapshot() { return core().snapshot(); }

} // namespace ui
