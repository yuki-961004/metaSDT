#ifndef MODIFY_PARAMS_HPP
#define MODIFY_PARAMS_HPP

#include <string>
#include <vector>
#include <unordered_map>

// 定义一个结构体来模拟 R 中的 list (包含 free, fixed, constant)
struct ParamGroup {
    std::unordered_map<std::string, std::vector<double>> free;
    std::unordered_map<std::string, std::vector<double>> fixed;
    std::unordered_map<std::string, std::vector<double>> constant;
};

// 声明：14 个元认知模型的全局通用参数列表 (C++ 版)
ParamGroup default_params();

// 声明：全局预设的模型参数边界
std::unordered_map<std::string, std::vector<double>> default_bounds();

// 声明：用于返回结果的结构体，包含拍扁后的字典以及分类保留的结构化参数
struct ModifiedParamsResult {
    std::unordered_map<std::string, std::vector<double>> flat;
    ParamGroup structured;
    std::vector<std::string> name_free;
    std::vector<std::string> name_fixed;
    std::vector<std::string> name_constant;
    std::vector<double> lower_bounds; // 自动生成的自由参数下界 (拍扁为一维)
    std::vector<double> upper_bounds; // 自动生成的自由参数上界 (拍扁为一维)
    int numb_free;
    int numb_fixed;
    int numb_constant;
};

// 声明：核心处理函数
ModifiedParamsResult modify_params(
    const ParamGroup& user_params,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower = {},
    const std::unordered_map<std::string, std::vector<double>>& custom_upper = {}
);

#endif // MODIFY_PARAMS_HPP