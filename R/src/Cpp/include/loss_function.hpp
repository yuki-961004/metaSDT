#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP

#include <vector>
#include <string>
#include <unordered_map>

// 定义返回结果：包含极大似然估计和模型比较所需的所有关键指标
struct LossResult {
    double logL; // 原始对数似然值
    double nll;  // 负对数似然 (用于 NLOPT 最小化)
    int k;       // 自由参数个数
    double aic;  // 赤池信息量准则
    double bic;  // 贝叶斯信息量准则
};

LossResult loss_function(
    const std::vector<std::vector<double>>& mult_mat,
    const std::vector<std::vector<double>>& freq_mat,
    int k
);

#endif // LOSS_FUNCTION_HPP