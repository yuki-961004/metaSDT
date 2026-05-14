#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "build_objective.hpp"

//// ========================================================== ////
//// NLopt 优化器与控制参数结构体声明
//// ========================================================== ////

// [Slot: NLoptControl]
// 控制 NLopt 优化器行为及多线程/进度显示的超参数结构体
struct NLoptControl {
    // 基础优化器控制参数
    std::string algorithm = "LN_BOBYQA"; // 使用的 nlopt 优化算法
    double xtol_rel = 1e-6;              // 相对参数容限
    int maxeval = 10000;                 // 最大函数计算次数
    double ftol_rel = 0.0;               // 相对函数容限
    double ftol_abs = 0.0;               // 绝对函数容限
    double xtol_abs = 0.0;               // 绝对参数容限
    double maxtime = 0.0;                // 最大运行时间
    double stopval = 0.0;                // 停止目标值
    int population = 0;                  // 种群大小 (用于全局启发式)
    double initial_step = 0.0;           // 初始步长

    // 局部优化器参数 (MLSL/AUGLAG 等算法使用)
    std::string local_algorithm = "LN_BOBYQA"; 

    // 高级控制参数
    std::vector<double> x_weights;       // 参数权重
    long seed = 1004;                    // 随机种子
    std::unordered_map<std::string, double> nlopt_params; // 其他传入 nlopt 的附加控制
    int vector_storage = 0;              // L-BFGS 等使用的向量存储，0 表示使用 nlopt 启发式默认值

    // 输出与多线程控制参数
    int print_level = 1;                 // 打印级别控制，0表示静默
    int threads = 0;                     // 线程数，<=0 表示使用所有可用 CPU 线程
    
    // 进度条控制参数
    std::string progress = "dynamic";    // 进度条模式：auto | dynamic | line | silent
    int progress_refresh_ms = 100;       // 刷新毫秒间隔
    double progress_line_interval_sec = 2.0; // 基于行的日志输出时间间隔
    double progress_line_interval_pct = 5.0; // 基于行的日志输出百分比间隔

    // EM-MAP 特定控制参数 (仅 estimate_map 使用)
    int em_max_iter = 100;               // EM 最大迭代次数
    double em_tol = 1e-3;                // EM 停止容限
    int em_patience = 0;                 // EM 收敛耐心值，0 表示禁用
    bool em_init_mle = true;             // 是否使用 MLE 初始化 EM 算法
};

//// ========================================================== ////
//// 单个被试/实验条件 拟合结果结构体声明
//// ========================================================== ////

// [Slot: SubjectFitResult]
// 用于存储并返回个体数据参数估计后的所有有效指标
struct SubjectFitResult {
    double subid = 0.0;                  // 被试唯一 ID
    std::string cond;                    // 实验条件
    
    // 拟合指标统计
    double logL = 0.0;                   // 对数似然值
    double logPrior = 0.0;               // 对数先验概率 (主要用于 MAP)
    double logPost = 0.0;                // 对数后验概率 (主要用于 MAP)
    double aic = 0.0;                    // 赤池信息量准则
    double bic = 0.0;                    // 贝叶斯信息量准则
    
    // 最佳参数结果
    std::unordered_map<std::string, std::vector<double>> best_params; // 结构化的最佳参数字典
    
    // 优化器运行状态
    int status = -1;                     // nlopt 运行返回状态码
    int n_evals = 0;                     // 函数求值总次数
    double optimum_value = 0.0;          // 优化到达的极小值
    std::string result_message;          // 优化结果人类可读提示
    std::string stop_reason;             // 优化停止的简短原因
};

//// ========================================================== ////
//// 核心业务逻辑函数声明
//// ========================================================== ////

// [Slot: estimate_mle_declaration]
// 执行最大似然估计的核心暴露接口
std::vector<SubjectFitResult> estimate_mle(
    const std::unordered_map<std::string, std::vector<double>>& df,           // DataFrame 映射表
    const std::unordered_map<std::string, std::string>& colnames,             // 列名映射表
    const ParamGroup& user_params,                                            // 用户自定义参数
    const std::string& model_name,                                            // 模型名称
    const NLoptControl& control,                                              // nlopt 控制参数结构
    const std::unordered_map<std::string, std::vector<double>>& custom_lower, // 自定义下界
    const std::unordered_map<std::string, std::vector<double>>& custom_upper  // 自定义上界
);
