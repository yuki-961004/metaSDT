#include "../include/modify_params.hpp"
#include <iostream>

// 14 个元认知模型的全局通用参数列表 (C++ 版)
ParamGroup generate_default_params() {
    ParamGroup defaults;

    // ==========================================================================
    // 【自由参数 (Free)】
    // 默认参与 NLOPT 优化的参数。这些是描述人类最基础行为表现的核心指标。
    // ==========================================================================
    defaults.free["d"] = {1.5};                     // [All models] 信号与噪声的距离 (敏感度 d')
    defaults.free["c_resp"] = {0.0};                // [All models] 一阶决策标准 (反应偏好 c)

    // ==========================================================================
    // 【固定参数 (Fixed)】
    // 具有心理学/数学意义，但默认不参与优化的参数。
    // (注: 在实际拟合含有置信度的实验时，用户通常会用 modifyList 将 c_conf 移入 free)
    // ==========================================================================
    // --- 感知层基础参数 (Sensory/Type-1) ---
    defaults.fixed["sd_signal"] = {1.0};            // [All models] 信号分布的标准差
    defaults.fixed["sd_noise"] = {1.0};             // [All models] 噪声分布的标准差
    
    // --- 置信度核心参数 (Meta/Type-2) ---
    defaults.fixed["c_conf"] = {};                  // [All models] 二阶置信度边界向量 (默认无置信度，为空)
    
    // --- 变异模型专属参数 (按机制分类) ---
    // 1. 噪声机制 (Noise)
    defaults.fixed["sd_meta"] = {0.5};              // [Gauss, LogN, WEV, CASANDRE] 元认知层面的额外标准差
    defaults.fixed["sd_c_resp"] = {0.1};            // [SDRM] 一阶决策标准本身的波动噪声
    defaults.fixed["sd_c_conf"] = {0.1};            // [SDRM, SOC] 二阶置信度标准本身的波动噪声
    
    // 2. 信号损失与时间累积 (Dynamics)
    defaults.fixed["rate_decay"] = {1.0};           // [Decay] 信号传导至元认知层的保留比率
    defaults.fixed["rate_accum"] = {0.1};           // [Post-Dec, 2DSD] 决策后继续累积证据的速率或时间
    
    // 3. 启发式与通道权重 (Heuristics & Channels)
    defaults.fixed["w_pos_evidence"] = {1.0};       // [PE-Flex] 积极证据(与决策一致的证据)的权重 (注: PE模型强制为1)
    defaults.fixed["w_visibility"] = {0.5};         // [WEV] 刺激可见度(全局难度)对信心的影响权重
    defaults.fixed["w_unconscious"] = {0.5};        // [DC] 无意识通道能获取到的信号权重
    
    // 4. 变量关联性 (Correlation)
    defaults.fixed["rho_sens_meta"] = {0.5};        // [SDRM, SOC] 一阶感知证据与二阶信心证据的皮尔逊相关系数

    // ==========================================================================
    // 【常量参数 (Constant)】
    // 纯粹控制计算机模拟器行为的超参数，永远不参与拟合。
    // ==========================================================================
    defaults.constant["sim_trials"] = {100000.0};   // [DC, Post-Dec, SOC 等无解析解模型] 蒙特卡洛模拟的试次数
    defaults.constant["rate_lapse"] = {0.0};        // [CASANDRE 等] 随机猜测/按错键的概率 (Lapse Rate)
    defaults.constant["calc_tol"] = {1e-10};        // [All models] 极小值容差，防止对数似然函数中出现 log(0)
    defaults.constant["rng_seed"] = {42.0};         // [模拟型模型] 随机数种子，确保优化器在相同参数下得到相同结果

    return defaults;
}

// 核心处理函数
ModifiedParamsResult modify_and_flatten_params(const ParamGroup& user_params) {
    
    // 1. 初始化默认值
    // 从全局默认配置表中加载，代替硬编码
    ParamGroup params = generate_default_params();

    // 【新增规则：以用户输入为最高优先级】
    // 如果用户在 free 中显式指定了参数（例如输入了 list(d=2.0, c_conf=...)），
    // 那么系统默认的 free 列表里未被用户明确提及的参数（如 c_resp），
    // 将自动降级为 fixed 参数，从而不再作为自由参数被优化。
    if (!user_params.free.empty()) {
        std::vector<std::string> to_demote;
        for (const auto& kv : params.free) {
            if (user_params.free.find(kv.first) == user_params.free.end()) {
                to_demote.push_back(kv.first);
            }
        }
        for (const auto& key : to_demote) {
            params.fixed[key] = params.free[key];
            params.free.erase(key);
        }
    }

    // 2. 动态覆盖与互斥清理 (按 constant -> fixed -> free 的优先级升序处理)
    // 这种做法能够安全响应用户的任何层级移动 (例如把默认 free 的参数放入 fixed 固定住)。
    // 即使同一参数被用户错误地放置在了多个槽位，最后遍历的高优先级也会覆盖并清理低优先级。
    
    for (const auto& kv : user_params.constant) {
        params.constant[kv.first] = kv.second;
        params.free.erase(kv.first);
        params.fixed.erase(kv.first);
    }

    for (const auto& kv : user_params.fixed) {
        params.fixed[kv.first] = kv.second;
        params.free.erase(kv.first);
        params.constant.erase(kv.first);
    }

    for (const auto& kv : user_params.free) {
        params.free[kv.first] = kv.second;
        params.fixed.erase(kv.first);
        params.constant.erase(kv.first);
    }

    // 3. 将整个 params 拍扁 (Flatten) 为一个简单的一维字典
    std::unordered_map<std::string, std::vector<double>> flat_params;
    
    // 按照 constant -> fixed -> free 的顺序插入
    // 因为前面已经完成了互斥清理（erase），所以这里直接合并不会发生重复碰撞
    for (const auto& kv : params.constant) {
        flat_params[kv.first] = kv.second;
    }
    for (const auto& kv : params.fixed) {
        flat_params[kv.first] = kv.second;
    }
    for (const auto& kv : params.free) {
        flat_params[kv.first] = kv.second;
    }

    // 返回结构体：既包含拍扁后的字典，也包含更新后的 ParamGroup 结构化信息
    ModifiedParamsResult result;
    result.flat = flat_params;
    result.structured = params;
    
    return result;
}
