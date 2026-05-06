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
ParamGroup generate_default_params();

// 声明：核心处理函数
std::unordered_map<std::string, std::vector<double>> modify_and_flatten_params(const ParamGroup& user_params);

#endif // MODIFY_PARAMS_HPP