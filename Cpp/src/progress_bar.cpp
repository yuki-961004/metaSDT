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
#include <thread>

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

/* ========================================================================== *
 *                              Internal Helpers                              *
 * ========================================================================== */

// 将字符串转换为小写, 以便进行宽容的模式匹配
std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

// 当环境变量存在且非空时返回 true
bool env_true(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::string(v).size() > 0;
}

/* ========================================================================== *
 *                         Progress Core (Thread-Safe)                        *
 * ========================================================================== */

class ProgressCore {
public:
    ~ProgressCore() {
        stop_render_thread();
    }

/* -------------------------------------------------------------------------- *
 *                        Lifecycle & Initialization                          *
 * -------------------------------------------------------------------------- */

    // 启动一个新的进度会话, 并重置所有计数器和时间戳
    void start(
        std::size_t total,
        const std::string& title,
        int refresh_ms,
        const ProgressOptions& options
    ) {
        // 在重用之前, 确保之前的渲染线程已彻底停止, 避免多线程冲突
        stop_render_thread();

        {
            // 加锁保护, 安全地初始化进度条的内部状态
            std::lock_guard<std::mutex> lock(mu_);
            total_ = total;
            current_ = 0;
            title_ = title.empty() ? "Progress" : title;
            opts_ = options;
            if (refresh_ms > 0) {
                opts_.refresh_ms = refresh_ms;
            }
            resolved_mode_ = resolve_mode(opts_.mode);
            running_ = true;
            finished_ = false;
            stop_worker_ = false;
            started_at_ = std::chrono::steady_clock::now();
            last_emit_at_ = started_at_;
            last_emit_pct_ = -1.0;
        }

        // 除非是静默模式, 否则启动后台轮询线程来周期性地渲染进度条
        if (resolved_mode_ != "silent") {
            worker_ = std::thread(&ProgressCore::worker_loop, this);
        }
    }

/* -------------------------------------------------------------------------- *
 *                           State Update Methods                           *
 * -------------------------------------------------------------------------- */

    // 直接设置当前的进度值(绝对值更新)
    void set(std::size_t current) {
        std::lock_guard<std::mutex> lock(mu_);
        current_ = current;
        // 确保进度不超出最大值
        if (total_ > 0 && current_ > total_) {
            current_ = total_;
        }
    }

    // 在当前进度的基础上步进(增加)指定的数量
    void advance(std::size_t step) {
        std::lock_guard<std::mutex> lock(mu_);
        current_ += step;
        // 确保进度不超出最大值
        if (total_ > 0 && current_ > total_) {
            current_ = total_;
        }
    }

/* -------------------------------------------------------------------------- *
 *                       Finish & Snapshot Utilities                        *
 * -------------------------------------------------------------------------- */

    // 标记会话已完成, 强制进行最后一次渲染, 然后停止工作线程
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!running_) {
                return;
            }
            finished_ = true;
            running_ = false;
            stop_worker_ = true;
        }

        // 强制渲染最终进度条状态(比如达到 100% 或被外部中断)
        render_once(true);
        stop_render_thread();

        // 如果是动态单行刷新模式, 最后补一个换行符, 防止后续输出覆盖进度条
        if (resolved_mode_ == "dynamic") {
            std::lock_guard<std::mutex> lk(out_mu_);
            std::cout << "\n";
            std::cout.flush();
        }
    }

    // 返回当前进度的线程安全快照, 以便进行监控或外部程序的检查
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
        s.percent = (total_ > 0)
            ? (100.0 * static_cast<double>(current_) /
               static_cast<double>(total_))
            : 0.0;
        s.speed = (s.elapsed_sec > 1e-9)
            ? (static_cast<double>(current_) / s.elapsed_sec)
            : 0.0;
        s.eta_sec = (s.speed > 1e-9 && total_ > current_)
            ? (static_cast<double>(total_ - current_) / s.speed)
            : 0.0;
        s.started = started_at_.time_since_epoch().count() != 0;
        s.running = running_;
        s.finished = finished_;
        s.requested_mode = opts_.mode;
        s.resolved_mode = resolved_mode_;
        return s;
    }

private:
/* -------------------------------------------------------------------------- *
 *                      Display Mode Resolution Logic                       *
 * -------------------------------------------------------------------------- */

    // 根据请求的模式和当前系统/IDE 环境的能力, 解析出最终的输出模式
    std::string resolve_mode(const std::string& requested) {
        std::string m = to_lower(requested);
        // 如果用户明确请求了特定的已知模式, 则直接应用
        if (m == "dynamic" || m == "line" || m == "silent") {
            return m;
        }

        // 检查当前标准输出是否连接到真实的物理终端 (TTY)
        bool is_tty = UI_ISATTY(UI_FILENO(stdout)) != 0;
        // 检查是否处于不支持 "\r" 动态刷新的 IDE 环境 (如 VSCode, Positron)
        bool in_vscode_like = env_true("VSCODE_PID") ||
            env_true("TERM_PROGRAM") ||
            env_true("POSITRON_VERSION") ||
            env_true("POSITRON_SESSION_ID");
        // 检查是否处于 Jupyter Notebook / JupyterHub 浏览器环境
        bool in_jupyter = env_true("JPY_PARENT_PID") ||
            env_true("IPYKERNEL_PARENT_PID") ||
            env_true("JUPYTERHUB_USER");
        // 检查是否处于 Quarto 后台渲染进程中
        bool in_quarto = env_true("QUARTO_PROJECT_ROOT") ||
            env_true("QUARTO_PROFILE") ||
            env_true("QUARTO_RENDER");

        // 综合判断: 如果处于上述任何受限环境, 或者不是真实终端, 则回退到逐行输出模式
        if (in_jupyter || in_quarto || in_vscode_like || !is_tty) {
            return "line";
        }
        // 否则默认使用体验更好的动态重绘模式
        return "dynamic";
    }

/* -------------------------------------------------------------------------- *
 *                    Background Polling & Render Logic                     *
 * -------------------------------------------------------------------------- */

    // 后台工作线程循环: 主动按设定频率唤醒并触发进度渲染
    void worker_loop() {
        while (true) {
            int sleep_ms = 100;
            bool should_stop = false;

            {
                // 获取锁以安全读取刷新间隔和停止标志
                std::lock_guard<std::mutex> lock(mu_);
                sleep_ms = (opts_.refresh_ms > 0) ? opts_.refresh_ms : 100;
                should_stop = stop_worker_;
            }

            if (should_stop) {
                break;
            }

            // 休眠指定的毫秒数, 避免过度占用 CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            // 尝试执行常规渲染(非强制)
            render_once(false);
        }
    }

    // 安全地发出停止信号, 并汇合 (join) 渲染线程
    void stop_render_thread() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_worker_ = true;
        }
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    // 核心渲染逻辑. `force=true` 会绕过时间或进度的节流(throttling)限制, 强制渲染
    void render_once(bool force) {
        ProgressSnapshot s = snapshot();
        // 如果解析结果是静默模式, 则直接退出不进行任何打印
        if (s.resolved_mode == "silent") {
            return;
        }

        auto now = std::chrono::steady_clock::now();
        bool emit = force;
        std::chrono::steady_clock::time_point last_emit_at;
        double last_emit_pct = -1.0;
        ProgressOptions local_opts;

        {
            std::lock_guard<std::mutex> lock(mu_);
            last_emit_at = last_emit_at_;
            last_emit_pct = last_emit_pct_;
            local_opts = opts_;
        }

        if (!emit) {
            // 读取阈值配置, 设定逐行模式下触发输出的最小时间间隔和进度增量百分比
            const double sec_step = local_opts.line_interval_sec > 0.0
                ? local_opts.line_interval_sec
                : 2.0;
            const double pct_step = local_opts.line_interval_pct > 0.0
                ? local_opts.line_interval_pct
                : 5.0;
            
            // 计算自上次渲染以来流逝的时间
            const double elapsed_since_emit =
                std::chrono::duration<double>(now - last_emit_at).count();

            if (s.resolved_mode == "dynamic") {
                // 动态模式: 严格按设定的时间频率刷新 (默认 100ms)
                emit = elapsed_since_emit >=
                    (static_cast<double>(
                        local_opts.refresh_ms > 0 ?
                        local_opts.refresh_ms : 100
                    ) / 1000.0);
            } else {
                // 逐行输出模式 (line mode): 当时间跨度或进度跨度达到阈值时才输出, 避免刷屏太快
                emit = (elapsed_since_emit >= sec_step) ||
                    (s.percent >= last_emit_pct + pct_step);
            }
        }

        if (!emit) {
            // 未达到触发条件, 跳过本次渲染
            return;
        }

        std::ostringstream oss;
        if (s.resolved_mode == "dynamic") {
            // 动态模式: 使用回车符 '\r' 回到行首以覆盖上一行的内容
            const int bar_width = 24;
            // 计算进度条实际需要填充的字符数('=')
            const int filled = (s.total > 0)
                ? static_cast<int>(std::round(
                    (static_cast<double>(s.completed) /
                     static_cast<double>(s.total)) * bar_width
                  ))
                : 0;

            oss << "\r" << s.title << " [";
            for (int i = 0; i < bar_width; ++i) {
                oss << (i < filled ? '=' : ' ');
            }
            oss << "] "
                << std::fixed << std::setprecision(1) << s.percent << "% "
                << s.completed << "/" << s.total
                << " | elapsed " << std::setprecision(1)
                << s.elapsed_sec << "s"
                << " | eta " << std::setprecision(1) << s.eta_sec << "s"
                << " | speed " << std::setprecision(1) << s.speed << "/s";
            if (force || s.finished) {
                oss << " | done";
            }

            // 构造输出字符串, 使用加锁保护 iostream 输出避免多线程文字穿插
            std::lock_guard<std::mutex> lk(out_mu_);
            std::cout << oss.str();
            std::cout.flush();
        } else {
            // 逐行模式: 每次渲染生成新的一行, 提供最大的环境兼容性
            oss << s.title << ": "
                << std::fixed << std::setprecision(1) << s.percent << "%"
                << " | " << s.completed << "/" << s.total
                << " | elapsed " << std::setprecision(1)
                << s.elapsed_sec << "s"
                << " | eta " << std::setprecision(1) << s.eta_sec << "s"
                << " | speed " << std::setprecision(1) << s.speed << "/s";
            if (force || s.finished) {
                oss << " | done";
            }

            std::lock_guard<std::mutex> lk(out_mu_);
            std::cout << oss.str() << "\n";
            std::cout.flush();
        }

        {
            // 更新上次渲染的状态记录
            std::lock_guard<std::mutex> lock(mu_);
            last_emit_at_ = now;
            last_emit_pct_ = s.percent;
        }
    }

private:
    mutable std::mutex mu_;
    std::mutex out_mu_;
    std::thread worker_;

    std::size_t total_ = 0;
    std::size_t current_ = 0;
    std::string title_ = "Progress";
    ProgressOptions opts_{};
    std::string resolved_mode_ = "silent";
    bool running_ = false;
    bool finished_ = false;
    bool stop_worker_ = true;
    std::chrono::steady_clock::time_point started_at_{};
    std::chrono::steady_clock::time_point last_emit_at_{};
    double last_emit_pct_ = -1.0;
};

// 进程全局单例, 供暴露给外部的包装 API 使用
ProgressCore& core() {
    static ProgressCore instance;
    return instance;
}

} // namespace

/* ========================================================================== *
 *                              Public API Wrappers                           *
 * ========================================================================== */

void progress_start(
    std::size_t total,
    const std::string& title,
    int refresh_ms,
    const ProgressOptions& options
) {
    core().start(total, title, refresh_ms, options);
}

void progress_set(std::size_t current) { core().set(current); }
void progress_advance(std::size_t step) { core().advance(step); }
void progress_finish() { core().finish(); }
ProgressSnapshot progress_last_snapshot() { return core().snapshot(); }

} // namespace ui
