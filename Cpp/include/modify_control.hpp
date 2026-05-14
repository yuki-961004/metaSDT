#pragma once

#include <string>
#include "estimate_mle.hpp"

/* ========================================================================== *
 *                            Stan Control Settings                           *
 * ========================================================================== */

// 控制 Stan Math 驱动的 HMC 采样行为
struct StanControl {
    std::string algorithm = "nuts";   // 采样器类型: nuts 或 static_hmc
    int chains = 4;                  // 独立 Markov chains 数量
    int warmup = 1000;               // 每条链丢弃的 warmup 迭代数
    int samples = 1000;              // 每条链保留的 posterior draws 数量
    int thin = 1;                    // 采样间隔, 1 表示保留每次采样

    double step_size = 0.05;         // HMC leapfrog 初始步长
    int leapfrog_steps = 10;         // 每次 proposal 的 leapfrog 步数
    int max_tree_depth = 8;          // NUTS 每次采样允许的最大树深度
    bool adapt_step_size = true;     // 是否在 warmup 阶段自适应步长
    double target_accept = 0.80;     // warmup 步长调整的目标接受率
    double min_step_size = 1e-6;     // 防止步长收缩到数值零
    double max_step_size = 1.0;      // 防止步长爆炸导致 proposal 失败
    double max_delta_energy = 1000.0;// NUTS 发散判定能量阈值
    double initial_jitter = 0.05;    // 多链初始点的轻微扰动尺度

    long seed = 1004;                // 采样随机种子
    int print_level = 1;             // 0 表示静默, >0 表示显示进度
    int threads = 0;                 // <=0 表示使用 OpenMP 默认线程数

    std::string progress = "dynamic";
    int progress_refresh_ms = 100;
    double progress_line_interval_sec = 2.0;
    double progress_line_interval_pct = 5.0;
};

// Normalize control values and inject estimator-specific defaults.
NLoptControl modify_control(
    const NLoptControl& input,
    const std::string& estimator
);

StanControl modify_control(
    const StanControl& input,
    const std::string& estimator
);
