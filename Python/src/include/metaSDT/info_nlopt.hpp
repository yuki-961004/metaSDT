#pragma once

#include <nlopt.hpp>

#include <string>

// ============================================================================
// NLopt 信息模块
// ============================================================================
// 设计目标:
// 1. 统一管理 NLopt 返回码在项目内的字符串映射逻辑
// 2. 给 MAP 与 MLE 提供一致的状态分类语义, 避免重复代码
// 3. 预留更丰富的诊断字段, 便于后续扩展日志和调试输出

struct NLoptStatusInfo {
    int code = 0;
    std::string code_name = "UNKNOWN";
    std::string message = "Unknown result";
    std::string stop_reason = "failure";
    bool is_success = false;
    bool is_stopping_condition = false;
    bool is_error = false;
};

// ============================================================================
// 基础映射接口
// ============================================================================
std::string nlopt_result_message(nlopt::result code);
std::string nlopt_stop_reason(nlopt::result code);
std::string nlopt_result_code_name(nlopt::result code);

// ============================================================================
// 状态分类接口
// ============================================================================
bool nlopt_is_success(nlopt::result code);
bool nlopt_is_stopping_condition(nlopt::result code);
bool nlopt_is_error(nlopt::result code);

// ============================================================================
// 汇总接口
// ============================================================================
NLoptStatusInfo nlopt_status_info(nlopt::result code);

// ============================================================================
// 诊断字符串接口
// ============================================================================
// 返回格式示例:
// code=4,name=MAXEVAL_REACHED,reason=maxeval,message=Maximum evaluations reached
std::string nlopt_status_summary(nlopt::result code);
