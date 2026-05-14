#include "../include/info_nlopt.hpp"

#include <string>

/* ========================================================================== *
 *                    NLopt Status and Diagnostics Module                     *
 * ========================================================================== */

// 将 NLopt 的结果码映射为面向用户的可读消息
// 该消息用于日志与结果表展示, 强调"发生了什么"
std::string nlopt_result_message(nlopt::result code) {
    switch (code) {
        case nlopt::SUCCESS:
            return "Success";
        case nlopt::STOPVAL_REACHED:
            return "Stop value reached";
        case nlopt::FTOL_REACHED:
            return "Function tolerance reached";
        case nlopt::XTOL_REACHED:
            return "Parameter tolerance reached";
        case nlopt::MAXEVAL_REACHED:
            return "Maximum evaluations reached";
        case nlopt::MAXTIME_REACHED:
            return "Maximum time reached";
        case nlopt::FAILURE:
            return "Generic failure";
        case nlopt::INVALID_ARGS:
            return "Invalid arguments";
        case nlopt::OUT_OF_MEMORY:
            return "Out of memory";
        case nlopt::ROUNDOFF_LIMITED:
            return "Roundoff limited progress";
        case nlopt::FORCED_STOP:
            return "Forced stop";
        default:
            return "Unknown result";
    }
}

// 将 NLopt 结果码映射为项目内部统一的 stop_reason 标签
// 该标签用于跨模块统计(MLE/MAP)与下游程序化判断
std::string nlopt_stop_reason(nlopt::result code) {
    switch (code) {
        case nlopt::STOPVAL_REACHED:
            return "stopval";
        case nlopt::FTOL_REACHED:
            return "ftol";
        case nlopt::XTOL_REACHED:
            return "xtol";
        case nlopt::MAXEVAL_REACHED:
            return "maxeval";
        case nlopt::MAXTIME_REACHED:
            return "maxtime";
        case nlopt::SUCCESS:
            return "success";
        case nlopt::ROUNDOFF_LIMITED:
            return "roundoff_limited";
        case nlopt::FORCED_STOP:
            return "forced_stop";
        default:
            return "failure";
    }
}

// 将 NLopt 结果码映射为稳定的枚举名字符串
// 相比 message, 此字段更适合机器检索与问题定位
std::string nlopt_result_code_name(nlopt::result code) {
    switch (code) {
        case nlopt::SUCCESS:
            return "SUCCESS";
        case nlopt::STOPVAL_REACHED:
            return "STOPVAL_REACHED";
        case nlopt::FTOL_REACHED:
            return "FTOL_REACHED";
        case nlopt::XTOL_REACHED:
            return "XTOL_REACHED";
        case nlopt::MAXEVAL_REACHED:
            return "MAXEVAL_REACHED";
        case nlopt::MAXTIME_REACHED:
            return "MAXTIME_REACHED";
        case nlopt::FAILURE:
            return "FAILURE";
        case nlopt::INVALID_ARGS:
            return "INVALID_ARGS";
        case nlopt::OUT_OF_MEMORY:
            return "OUT_OF_MEMORY";
        case nlopt::ROUNDOFF_LIMITED:
            return "ROUNDOFF_LIMITED";
        case nlopt::FORCED_STOP:
            return "FORCED_STOP";
        default:
            return "UNKNOWN";
    }
}

// NLopt 约定: 正值表示成功终止
bool nlopt_is_success(nlopt::result code) {
    return static_cast<int>(code) > 0;
}

// 判断是否属于"达到停止条件"的终止
// 包括容差收敛, 预算耗尽和强制停止等非致命结束路径
bool nlopt_is_stopping_condition(nlopt::result code) {
    switch (code) {
        case nlopt::STOPVAL_REACHED:
        case nlopt::FTOL_REACHED:
        case nlopt::XTOL_REACHED:
        case nlopt::MAXEVAL_REACHED:
        case nlopt::MAXTIME_REACHED:
        case nlopt::ROUNDOFF_LIMITED:
        case nlopt::FORCED_STOP:
            return true;
        default:
            return false;
    }
}

// NLopt 约定: 负值表示错误
bool nlopt_is_error(nlopt::result code) {
    return static_cast<int>(code) < 0;
}

// 汇总所有诊断字段, 构造结构化状态对象
// 该函数是外部模块获取 NLopt 状态信息的统一入口
NLoptStatusInfo nlopt_status_info(nlopt::result code) {
    NLoptStatusInfo out;
    out.code = static_cast<int>(code);
    out.code_name = nlopt_result_code_name(code);
    out.message = nlopt_result_message(code);
    out.stop_reason = nlopt_stop_reason(code);
    out.is_success = nlopt_is_success(code);
    out.is_stopping_condition = nlopt_is_stopping_condition(code);
    out.is_error = nlopt_is_error(code);
    return out;
}

// 生成单行状态摘要, 便于日志落盘与快速排障
// 格式示例: code=4,name=MAXEVAL_REACHED,reason=maxeval,message=...
std::string nlopt_status_summary(nlopt::result code) {
    const NLoptStatusInfo info = nlopt_status_info(code);
    return "code=" + std::to_string(info.code) +
           ",name=" + info.code_name +
           ",reason=" + info.stop_reason +
           ",message=" + info.message;
}
