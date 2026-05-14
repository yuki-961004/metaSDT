#ifndef CRITERION_LIKELIHOOD_HPP
#define CRITERION_LIKELIHOOD_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <stdexcept>

/* ========================================================================== *
 *                 Data Structure: Likelihood Result                          *
 * ========================================================================== */

// 用于在一次计算完成后, 打包返回所有常用的统计推断评价指标, 极其易于向 Python/R 输出
template <typename T>
struct LikelihoodResult {
    T logL;      // 原始对数似然值
    T nll;       // 负对数似然 (用于优化器最小化)
    int k;       // 自由参数个数
    T aic;       // 赤池信息量准则
    T bic;       // 贝叶斯信息量准则
};

/* ========================================================================== *
 *                   Core Function: Criterion Likelihood                      *
 * ========================================================================== */

// 接收模型输出的概率矩阵, 真实数据的频数矩阵, 计算对数似然与信息准则
// 如果提供了自由参数和对应的参数表, 它将自动激活隐藏的 L_n 正则化惩罚逻辑
template <typename T>
inline LikelihoodResult<T> criterion_likelihood(
    const std::vector<std::vector<std::vector<T>>>& mult_mat,
    const std::vector<std::vector<std::vector<double>>>& freq_mat,
    int k,
    const std::vector<T>& free_params = {},
    const std::unordered_map<std::string, std::vector<T>>& std_params = {}
) {
/* ========================================================================== *
 *                   1. Basic Safety Defenses and Validation                  *
 * ========================================================================== */

    // 确保传入的理论概率惩罚矩阵和真实观测的频数矩阵不为空, 防止后续迭代器发生致命越界
    if (mult_mat.empty() || freq_mat.empty()) {
        throw std::invalid_argument("Error: Input matrices cannot be empty.");
    }

/* ========================================================================== *
 *          2. Core Iteration: Calculate Total LogL and Trials (N)            *
 * ========================================================================== */

    // LogL 是频数 (Freq) 与对数概率 (Log-Prob) 的乘积之和
    // N 是实验中被试按键反应的总次数, 它是用于后续计算 BIC 指标的基石
    T logL = 0.0;
    double N = 0.0;

    for (size_t d = 0; d < mult_mat.size(); ++d) {
        for (size_t i = 0; i < mult_mat[d].size(); ++i) {
            for (size_t j = 0; j < mult_mat[d][i].size(); ++j) {
                logL += mult_mat[d][i][j];
                N += freq_mat[d][i][j];
            }
        }
    }

/* ========================================================================== *
 *                             3. Regularization                              *
 * ========================================================================== */

    T reg_term = 0.0;
    if (!free_params.empty() && !std_params.empty()) {
        auto it_L = std_params.find("L");
        auto it_pen = std_params.find("penalty");
        if (it_L != std_params.end() && !it_L->second.empty()) {
            double L_val = static_cast<double>(it_L->second[0]);
            double pen_val = 1.0;
            if (it_pen != std_params.end() && !it_pen->second.empty()) {
                pen_val = static_cast<double>(it_pen->second[0]);
            }
            
            using std::pow;
            using std::abs;
            for (T val : free_params) {
                if (L_val == 0.0) {
                    reg_term += 1.0; // 防止 pow(0,0) 的未定义行为
                } else if (L_val == 1.0) {
                    reg_term += abs(val); // L1 Lasso
                } else if (L_val == 2.0) {
                    reg_term += val * val; // L2 Ridge (优化运算速度)
                } else {
                    reg_term += pow(abs(val), L_val); // L_n
                }
            }
            reg_term *= pen_val;
        }
    }

/* ========================================================================== *
 *                 4. Information Criterion (AIC, BIC, NLL)                   *
 * ========================================================================== */

    // 将纯正的 logL 转换为更具统计意义的惩罚指标
    // 注意: 优化器 (如 NLopt 或 L-BFGS) 通常是寻找函数的"最小值", 
    // 因此我们将 logL 取反, 生成负对数似然 (NLL) 专门供它们使用
    LikelihoodResult<T> res;
    res.logL = logL;
    res.nll = -(logL - reg_term); // 供优化器使用的最小化目标
    res.k = k;
    res.aic = 2.0 * k - 2.0 * logL;
    
    using std::log; // 开启 ADL (参数依赖查找) 卫生规范, 防止 stan::math::var 在不同命名空间中迷失
    if (N > 0.0) {
        res.bic = k * log(N) - 2.0 * logL;
    } else {
        res.bic = 0.0; 
    }
    return res;
}

#endif // CRITERION_LIKELIHOOD_HPP