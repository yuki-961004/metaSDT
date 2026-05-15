#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "task_builder.hpp"

//// ========================================================== ////
//// NLopt 浼樺寲鍣ㄤ笌鎺у埗鍙傛暟缁撴瀯浣撳０鏄?
//// ========================================================== ////

// [Slot: NLoptControl]
// 鎺у埗 NLopt 浼樺寲鍣ㄨ涓哄強澶氱嚎绋?杩涘害鏄剧ず鐨勮秴鍙傛暟缁撴瀯浣?
struct NLoptControl {
    // 鍩虹浼樺寲鍣ㄦ帶鍒跺弬鏁?
    std::string algorithm = "LN_BOBYQA"; // 浣跨敤鐨?nlopt 浼樺寲绠楁硶
    double xtol_rel = 1e-6;              // 鐩稿鍙傛暟瀹归檺
    int maxeval = 10000;                 // 鏈€澶у嚱鏁拌绠楁鏁?
    double ftol_rel = 0.0;               // 鐩稿鍑芥暟瀹归檺
    double ftol_abs = 0.0;               // 缁濆鍑芥暟瀹归檺
    double xtol_abs = 0.0;               // 缁濆鍙傛暟瀹归檺
    double maxtime = 0.0;                // 鏈€澶ц繍琛屾椂闂?
    double stopval = 0.0;                // 鍋滄鐩爣鍊?
    int population = 0;                  // 绉嶇兢澶у皬 (鐢ㄤ簬鍏ㄥ眬鍚彂寮?
    double initial_step = 0.0;           // 鍒濆姝ラ暱

    // 灞€閮ㄤ紭鍖栧櫒鍙傛暟 (MLSL/AUGLAG 绛夌畻娉曚娇鐢?
    std::string local_algorithm = "LN_BOBYQA"; 

    // 楂樼骇鎺у埗鍙傛暟
    std::vector<double> x_weights;       // 鍙傛暟鏉冮噸
    long seed = 1004;                    // 闅忔満绉嶅瓙
    std::unordered_map<std::string, double> nlopt_params; // 鍏朵粬浼犲叆 nlopt 鐨勯檮鍔犳帶鍒?
    int vector_storage = 0;              // L-BFGS 绛変娇鐢ㄧ殑鍚戦噺瀛樺偍锛? 琛ㄧず浣跨敤 nlopt 鍚彂寮忛粯璁ゅ€?

    // 杈撳嚭涓庡绾跨▼鎺у埗鍙傛暟
    int print_level = 1;                 // 鎵撳嵃绾у埆鎺у埗锛?琛ㄧず闈欓粯
    int threads = 0;                     // 绾跨▼鏁帮紝<=0 琛ㄧず浣跨敤鎵€鏈夊彲鐢?CPU 绾跨▼
    
    // 杩涘害鏉℃帶鍒跺弬鏁?
    std::string progress = "dynamic";    // 杩涘害鏉℃ā寮忥細auto | dynamic | line | silent
    int progress_refresh_ms = 100;       // 鍒锋柊姣闂撮殧
    double progress_line_interval_sec = 2.0; // 鍩轰簬琛岀殑鏃ュ織杈撳嚭鏃堕棿闂撮殧
    double progress_line_interval_pct = 5.0; // 鍩轰簬琛岀殑鏃ュ織杈撳嚭鐧惧垎姣旈棿闅?

    // EM-MAP 鐗瑰畾鎺у埗鍙傛暟 (浠?estimate_map 浣跨敤)
    int em_max_iter = 100;               // EM 鏈€澶ц凯浠ｆ鏁?
    double em_tol = 1e-3;                // EM 鍋滄瀹归檺
    int em_patience = 0;                 // EM 鏀舵暃鑰愬績鍊硷紝0 琛ㄧず绂佺敤
    bool em_init_mle = true;             // 鏄惁浣跨敤 MLE 鍒濆鍖?EM 绠楁硶
};

//// ========================================================== ////
//// 鍗曚釜琚瘯/瀹為獙鏉′欢 鎷熷悎缁撴灉缁撴瀯浣撳０鏄?
//// ========================================================== ////

// [Slot: SubjectFitResult]
// 鐢ㄤ簬瀛樺偍骞惰繑鍥炰釜浣撴暟鎹弬鏁颁及璁″悗鐨勬墍鏈夋湁鏁堟寚鏍?
struct SubjectFitResult {
    double subid = 0.0;                  // 琚瘯鍞竴 ID
    std::string cond;                    // 瀹為獙鏉′欢
    
    // 鎷熷悎鎸囨爣缁熻
    double logL = 0.0;                   // 瀵规暟浼肩劧鍊?
    double logPrior = 0.0;               // 瀵规暟鍏堥獙姒傜巼 (涓昏鐢ㄤ簬 MAP)
    double logPost = 0.0;                // 瀵规暟鍚庨獙姒傜巼 (涓昏鐢ㄤ簬 MAP)
    double aic = 0.0;                    // 璧ゆ睜淇℃伅閲忓噯鍒?
    double bic = 0.0;                    // 璐濆彾鏂俊鎭噺鍑嗗垯
    
    // 鏈€浣冲弬鏁扮粨鏋?
    std::unordered_map<std::string, std::vector<double>> best_params; // 缁撴瀯鍖栫殑鏈€浣冲弬鏁板瓧鍏?
    
    // 浼樺寲鍣ㄨ繍琛岀姸鎬?
    int status = -1;                     // nlopt 杩愯杩斿洖鐘舵€佺爜
    int n_evals = 0;                     // 鍑芥暟姹傚€兼€绘鏁?
    double optimum_value = 0.0;          // 浼樺寲鍒拌揪鐨勬瀬灏忓€?
    std::string result_message;          // 浼樺寲缁撴灉浜虹被鍙鎻愮ず
    std::string stop_reason;             // 浼樺寲鍋滄鐨勭畝鐭師鍥?
};

//// ========================================================== ////
//// 鏍稿績涓氬姟閫昏緫鍑芥暟澹版槑
//// ========================================================== ////

// [Slot: estimate_mle_declaration]
// 鎵ц鏈€澶т技鐒朵及璁＄殑鏍稿績鏆撮湶鎺ュ彛
std::vector<SubjectFitResult> estimate_mle(
    const std::unordered_map<std::string, std::vector<double>>& df,           // DataFrame 鏄犲皠琛?
    const std::unordered_map<std::string, std::string>& colnames,             // 鍒楀悕鏄犲皠琛?
    const ParamGroup& user_params,                                            // 鐢ㄦ埛鑷畾涔夊弬鏁?
    const std::string& model_name,                                            // 妯″瀷鍚嶇О
    const NLoptControl& control,                                              // nlopt 鎺у埗鍙傛暟缁撴瀯
    const std::unordered_map<std::string, std::vector<double>>& custom_lower, // 鑷畾涔変笅鐣?
    const std::unordered_map<std::string, std::vector<double>>& custom_upper  // 鑷畾涔変笂鐣?
);
