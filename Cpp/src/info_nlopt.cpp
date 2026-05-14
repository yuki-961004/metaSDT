#include "../include/info_nlopt.hpp"

#include <string>

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

bool nlopt_is_success(nlopt::result code) {
    return static_cast<int>(code) > 0;
}

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

bool nlopt_is_error(nlopt::result code) {
    return static_cast<int>(code) < 0;
}

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

std::string nlopt_status_summary(nlopt::result code) {
    const NLoptStatusInfo info = nlopt_status_info(code);
    return "code=" + std::to_string(info.code) +
           ",name=" + info.code_name +
           ",reason=" + info.stop_reason +
           ",message=" + info.message;
}
