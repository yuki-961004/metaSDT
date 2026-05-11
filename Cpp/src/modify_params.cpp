#include "../include/modify_params.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// 14 个元认知模型的全局通用参数列表 (C++ 版)
ParamGroup default_params() {
    ParamGroup defaults;

    // ==========================================================================
    // 【自由参数 (Free)】
    // 默认参与 NLOPT 优化的参数。
    // 这些是描述人类最基础行为表现的核心指标。
    // ==========================================================================
    // [All models] 信号与噪声的距离 (敏感度 d')
    defaults.free["d"] = {1.5};
    // [All models] 一阶决策标准 (反应偏好 c)
    defaults.free["c_resp"] = {0.0};

    // ==========================================================================
    // 【固定参数 (Fixed)】
    // 具有心理学/数学意义，但默认不参与优化的参数。
    // (注: 在实际拟合含有置信度的实验时，
    // 用户通常会用 modifyList 将 c_conf 移入 free)
    // ==========================================================================
    // --- 感知层基础参数 (Sensory/Type-1) ---
    // [All models] 信号分布的标准差
    defaults.fixed["sd_signal"] = {1.0};
    // [All models] 噪声分布的标准差
    defaults.fixed["sd_noise"] = {1.0};
    
    // --- 置信度核心参数 (Meta/Type-2) ---
    // [All models] 二阶置信度边界向量 (默认无置信度，为空)
    defaults.fixed["c_conf"] = {};
    // [All models] 置信度判定标准总数 (默认由程序计算)
    defaults.fixed["n_conf"] = {};
    
    // --- 变异模型专属参数 (按机制分类) ---
    // 1. 噪声机制 (Noise)
    // [Gauss, LogN, WEV, CASANDRE] 元认知层面的额外标准差
    defaults.fixed["sd_meta"] = {0.5};
    // [SDRM] 一阶决策标准本身的波动噪声
    defaults.fixed["sd_c_resp"] = {0.1};
    // [SDRM, SOC] 二阶置信度标准本身的波动噪声
    defaults.fixed["sd_c_conf"] = {0.1};
    
    // 2. 信号损失与时间累积 (Dynamics)
    // [Decay] 信号传导至元认知层的保留比率
    defaults.fixed["rate_decay"] = {1.0};
    // [Post-Dec, 2DSD] 决策后继续累积证据的速率或时间
    defaults.fixed["rate_accum"] = {0.1};
    
    // 3. 启发式与通道权重 (Heuristics & Channels)
    // [PE-Flex] 积极证据的权重 (注: PE模型强制为1)
    defaults.fixed["w_pos_evidence"] = {1.0};
    // [WEV] 刺激可见度(全局难度)对信心的影响权重
    defaults.fixed["w_visibility"] = {0.5};
    // [DC] 无意识通道能获取到的信号权重
    defaults.fixed["w_unconscious"] = {0.5};
    
    // 4. 变量关联性 (Correlation)
    // [SDRM, SOC] 一阶感知与二阶信心证据的皮尔逊相关系数
    defaults.fixed["rho_sens_meta"] = {0.5};

    // ==========================================================================
    // 【常量参数 (Constant)】
    // 纯粹控制计算机模拟器行为的超参数，永远不参与拟合。
    // ==========================================================================
    // [DC, Post-Dec, SOC 等无解析解模型] 蒙特卡洛模拟的试次数
    defaults.constant["sim_trials"] = {100000.0};
    // [CASANDRE 等] 随机猜测/按错键的概率 (Lapse Rate)
    defaults.constant["rate_lapse"] = {0.0};
    // [All models] 极小值容差，防止对数似然函数中出现 log(0)
    defaults.constant["calc_tol"] = {1e-10};
    // [All models] Ln 正则化幂次 (默认空，不进行正则化)
    defaults.constant["L"] = {};
    // [All models] 正则化惩罚力度 (默认 1.0)
    defaults.constant["penalty"] = {1.0};
    // [模拟型模型] 随机数种子，确保优化器在相同参数下得到相同结果
    defaults.constant["rng_seed"] = {1004.0};

    return defaults;
}

// 全局预设的模型参数边界
std::unordered_map<std::string, std::vector<double>> default_bounds() {
    std::unordered_map<std::string, std::vector<double>> bounds;
    
    // 1. 核心感知层参数
    bounds["d"] = {-1.0, 10.0}; 
    bounds["c_resp"] = {-5.0, 5.0};
    bounds["sd_signal"] = {1e-4, 5.0}; 
    bounds["sd_noise"]  = {1e-4, 5.0};

    // 2. 置信度边界参数 (因采用相对步长，下界必须大于0以保证绝对坐标单调递增)
    bounds["c_conf"] = {1e-4, 20.0};

    // 3. 元噪声与波动参数 (必须大于0)
    bounds["sd_meta"] = {1e-3, 3.0}; 
    bounds["sd_c_resp"] = {0.01, 3.0};
    bounds["sd_c_conf"] = {0.001, 3.0};

    // 4. 权重与比率参数
    bounds["rate_decay"] = {0.0, 1.0};
    bounds["w_unconscious"] = {0.0, 1.0};
    bounds["w_visibility"] = {0.0, 1.0};
    bounds["w_pos_evidence"] = {0.51, 1.0};
    bounds["rho_sens_meta"] = {0.001, 0.999};

    // 5. 动态累积时间参数
    bounds["rate_accum"] = {1e-5, 3.0};

    return bounds;
}

// 核心处理函数
ModifiedParamsResult modify_params(
    const ParamGroup& user_params,
    const std::unordered_map<std::string, std::vector<double>>& custom_lower,
    const std::unordered_map<std::string, std::vector<double>>& custom_upper
) {
    
    // ==========================================================
    // 1. 初始化默认值与自由参数降级逻辑
    // ==========================================================
    // 从全局默认配置表中加载，代替硬编码
    ParamGroup params = default_params();

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

    // ==========================================================
    // 2. 动态覆盖与互斥清理
    // ==========================================================
    // 按 constant -> fixed -> free 的优先级升序处理。
    // 这种做法能够安全响应用户的任何层级移动 
    // (例如把默认 free 的参数放入 fixed 固定住)。
    // 即使同一参数被用户错误地放置在了多个槽位，
    // 最后遍历的高优先级也会覆盖并清理低优先级。
    
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

    // ==========================================================
    // 3. 字典扁平化 (Flatten)
    // ==========================================================
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

    // ==========================================================
    // 4. 边界关联与保守性检查 (处理 n_conf, c_conf 和 c_resp)
    // ==========================================================
    auto it_n_conf = flat_params.find("n_conf");
    auto it_c_conf = flat_params.find("c_conf");

    bool has_n_conf = (it_n_conf != flat_params.end() && 
                       !it_n_conf->second.empty());
    bool has_c_conf = (it_c_conf != flat_params.end() && 
                       !it_c_conf->second.empty());

    if (!has_n_conf) {
        if (has_c_conf) {
            // 没有设定 n_conf, 默认对称机制，自动计算为 2 * length(c_conf) + 1
            double calc_n_conf = 2.0 * it_c_conf->second.size() + 1.0;
            flat_params["n_conf"] = {calc_n_conf};
            params.fixed["n_conf"] = {calc_n_conf};
        }
    } else {
        // 用户设定了 n_conf，说明 c_conf 是完整的不等距分割点向量
        int n_conf_val = static_cast<int>(it_n_conf->second[0]);
        if (has_c_conf) {
            // 保守性检查：如果是奇数，则将中间值作为 c_resp
            if (n_conf_val % 2 != 0) {
                int mid_idx = n_conf_val / 2;
                if (mid_idx < static_cast<int>(it_c_conf->second.size())) {
                    double mid_val = it_c_conf->second[mid_idx];
                    auto it_c_resp = flat_params.find("c_resp");
                    if (it_c_resp != flat_params.end() && 
                        !it_c_resp->second.empty()) {
                        if (std::abs(it_c_resp->second[0] - mid_val) > 1e-6) {
                            std::cerr << "Warning: You set c_resp and n_conf, "
                                      << "but c_resp is not the median point "
                                      << "of c_conf. Please check if c_resp is "
                                      << "set correctly. The median point of "
                                      << "c_conf (" << mid_val << ") will be "
                                      << "used as c_resp for calculation.\n";
                        }
                    }
                    
                    // 强制更新 c_resp 以保持逻辑的一致性
                    flat_params["c_resp"] = {mid_val};
                    if (params.free.count("c_resp")) {
                        params.free["c_resp"] = {mid_val};
                    }
                    if (params.fixed.count("c_resp")) {
                        params.fixed["c_resp"] = {mid_val};
                    }
                    if (params.constant.count("c_resp")) {
                        params.constant["c_resp"] = {mid_val};
                    }
                }
            }
        }
    }

    // 返回结构体：既包含拍扁后的字典，也包含更新后的 ParamGroup 结构化信息
    ModifiedParamsResult result;
    result.flat = flat_params;
    result.structured = params;

    // 动态探测 c_conf 是否为绝对坐标向量 (以修复边界)
    bool is_full_vector = false;
    if (has_n_conf && has_c_conf) {
        int n_conf_val = static_cast<int>(it_n_conf->second[0]);
        if (n_conf_val == static_cast<int>(it_c_conf->second.size())) {
            is_full_vector = true;
        }
    }

    // ==========================================================
    // 5. 统计各类参数的名称和数量 (支持多维度向量，如 c_conf)
    // ==========================================================
    result.numb_free = 0;
    auto bounds_dict = default_bounds();
    
    // ==========================================================
    // 核心防御：使用完全动态的字母表排序 (Alphabetical Order)
    // 彻底消除 unordered_map 随机哈希带来的顺序错位，且无需硬编码任何参数名！
    // ==========================================================
    auto sort_keys = [](std::vector<std::string>& keys) {
        std::sort(keys.begin(), keys.end());
    };

    // 1. 处理 free 槽位
    std::vector<std::string> keys_free;
    for (const auto& kv : params.free) keys_free.push_back(kv.first);
    sort_keys(keys_free);
    
    for (const auto& key : keys_free) {
        result.name_free.push_back(key);
        const auto& val_vec = params.free.at(key);
        size_t p_size = val_vec.size();
        result.numb_free += p_size;
        
        double lb_base = -1e5, ub_base = 1e5;
        if (bounds_dict.count(key)) {
            lb_base = bounds_dict[key][0];
            ub_base = bounds_dict[key][1];
        }
        
        if (key == "c_conf" && is_full_vector) {
            lb_base = -10.0;
            ub_base = 10.0;
        }

        for (size_t i = 0; i < p_size; ++i) {
            double current_lb = lb_base;
            double current_ub = ub_base;
            
            if (custom_lower.count(key)) {
                if (custom_lower.at(key).size() > i) current_lb = custom_lower.at(key)[i];
                else if (custom_lower.at(key).size() == 1) current_lb = custom_lower.at(key)[0];
            }
            if (custom_upper.count(key)) {
                if (custom_upper.at(key).size() > i) current_ub = custom_upper.at(key)[i];
                else if (custom_upper.at(key).size() == 1) current_ub = custom_upper.at(key)[0];
            }

            result.lower_bounds.push_back(current_lb);
            result.upper_bounds.push_back(current_ub);
        }
    }

    // 2. 处理 fixed 槽位
    result.numb_fixed = 0;
    std::vector<std::string> keys_fixed;
    for (const auto& kv : params.fixed) keys_fixed.push_back(kv.first);
    sort_keys(keys_fixed);
    for (const auto& key : keys_fixed) {
        result.name_fixed.push_back(key);
        result.numb_fixed += params.fixed.at(key).size();
    }

    // 3. 处理 constant 槽位
    result.numb_constant = 0;
    std::vector<std::string> keys_constant;
    for (const auto& kv : params.constant) keys_constant.push_back(kv.first);
    sort_keys(keys_constant);
    for (const auto& key : keys_constant) {
        result.name_constant.push_back(key);
        result.numb_constant += params.constant.at(key).size();
    }
    
    return result;
}
