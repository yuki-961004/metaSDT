#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "matrix_freq.hpp"
#include "modify_params.hpp"

// 专门为每一个被试独立准备的拟合任务包 (Context Payload)
// 它是完美实现多线程和解耦的灵魂载体
struct SubjectFitTask {
    double subid;
    std::string model;                // 用户指定的模型名称
    MatrixFreq freq;                  // 该被试提前算好并缓存的二维频数矩阵
    ModifiedParamsResult params;      // 标准化后的参数字典与结构信息
};

// 任务工厂函数：接收全体数据，拆分被试，生成可供多核并行使用的独立任务包集合
std::vector<SubjectFitTask> build_fit_tasks(
    const std::unordered_map<std::string, std::vector<double>>& df,
    const std::unordered_map<std::string, std::string>& colnames,
    const ParamGroup& user_params,
    const std::string& model_name,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower = {},
    const std::unordered_map<std::string, std::vector<double>>& custom_upper = {}
);

// 供 NLOPT 直接调用的标准化静态目标函数
// 将会在多线程中被各自的 CPU 核心独立并发调用
double nll(
    const std::vector<double>& x, 
    std::vector<double>& grad, 
    void* f_data
);