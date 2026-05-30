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

    // === 新增：参数结构工具函数 ===
    
    // 将参数字典(如 flat 或 best_params)中的自由参数按 name_free 顺序抽取为一维向量
    std::vector<double> extract_free_vector(
        const std::unordered_map<std::string, std::vector<double>>& params_map
    ) const;

    // 将一维自由参数向量(如 nlopt 优化后结果)的值更新回参数字典
    void update_map_from_free_vector(
        std::unordered_map<std::string, std::vector<double>>& params_map, 
        const std::vector<double>& free_vec
    ) const;

    // 获取每个自由参数的维度大小(以 vector<int> 形式返回)
    std::vector<int> get_free_sizes() const;
};

// 声明：核心处理函数
ModifiedParamsResult modify_params(
    const ParamGroup& user_params,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower = {},
    const std::unordered_map<std::string, std::vector<double>>& custom_upper = {}
);

#endif // MODIFY_PARAMS_HPP