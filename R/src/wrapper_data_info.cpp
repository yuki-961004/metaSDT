#include <Rcpp.h>
#include "../../Cpp/include/data_info.hpp"

// 使用宏定义包住 include，骗过 Rcpp::sourceCpp 的正则检查
#define CORE_IMPL "../../Cpp/src/data_info.cpp"
#include CORE_IMPL

using namespace Rcpp;

//' Intelligently scan the dataset and extract subject-level information.
// [[Rcpp::export(name = "data_info")]]
List r_data_info(
    DataFrame df, 
    Nullable<List> colnames = R_NilValue
) {
    
    // 1. 将 DataFrame 转换为 C++ 的 unordered_map<string, vector<double>>
    std::unordered_map<std::string, std::vector<double>> cpp_df;
    CharacterVector df_names = df.names();
    
    for (int i = 0; i < df.size(); ++i) {
        // 只提取数值型或因子型列 (在 R 底层，factor 是 integer)
        // 心理学 SDT 实验的核心变量几乎全是数值/因子。若有纯字符串变量将被忽略。
        int type = TYPEOF(df[i]);
        if (type == INTSXP || type == REALSXP || type == LGLSXP) {
            cpp_df[as<std::string>(df_names[i])] = 
                as<std::vector<double>>(df[i]);
        }
    }
    
    // 2. 转换 colnames 字典
    std::unordered_map<std::string, std::string> cpp_colnames;
    if (colnames.isNotNull()) {
        List col_list(colnames);
        if (col_list.hasAttribute("names")) {
            CharacterVector c_names = col_list.names();
            for (int i = 0; i < col_list.size(); ++i) {
                cpp_colnames[as<std::string>(c_names[i])] = 
                    as<std::string>(col_list[i]);
            }
        }
    }
    
    // 4. 调用 C++ 核心函数进行闪电般的数据集扫描
    DataInfoResult res;
    try {
        res = ::data_info(/*df=*/cpp_df, /*colnames=*/cpp_colnames);
    } catch (std::exception& e) {
        stop(e.what());
    }
    
    // 新增：处理 C++ 返回的提示信息 (更轻量，可通过 suppressMessages 屏蔽)
    Rcpp::Function r_message("message");
    for (const auto& msg : res.messages) {
        r_message(msg);
    }
    
    // 5. 将结果完美转换回 R 结构
    List r_colnames;
    for (const auto& kv : res.colnames) {
        r_colnames[kv.first] = kv.second;
    }
    
    List r_subjects;
    for (const auto& kv : res.subjects) {
        double subid = kv.first;
        
        // 智能去除 double 的多余小数位并转换为 string 作为 list 的键 
        // (例如 1.0 变为 "1")
        std::string sub_key = std::to_string(subid);
        sub_key.erase(sub_key.find_last_not_of('0') + 1, std::string::npos);
        if (!sub_key.empty() && sub_key.back() == '.') {
            sub_key.pop_back();
        }
        
        const DataInfoSubject& subj = kv.second;
        
        // 转换行索引（核心：将 C++ 的 0 索引转换为 R 的 1 索引！）
        IntegerVector r_raw(subj.raw.size());
        for (size_t i = 0; i < subj.raw.size(); ++i) {
            r_raw[i] = subj.raw[i] + 1; 
        }
        
        // 转换分组后的行索引
        List r_condition;
        for (const auto& gkv : subj.condition) {
            IntegerVector g_idx(gkv.second.size());
            for (size_t i = 0; i < gkv.second.size(); ++i) {
                g_idx[i] = gkv.second[i] + 1;
            }
            r_condition[gkv.first] = g_idx;
        }
        
        List r_difficulty;
        for (const auto& dkv : subj.difficulty) {
            IntegerVector d_idx(dkv.second.size());
            for (size_t i = 0; i < dkv.second.size(); ++i) {
                d_idx[i] = dkv.second[i] + 1;
            }
            r_difficulty[dkv.first] = d_idx;
        }
        
        List r_info = List::create(
            Named("n_trials") = subj.info.n_trials,
            Named("n_blocks") = subj.info.n_blocks
        );
        
        r_subjects[sub_key] = List::create(
            Named("raw") = r_raw,
            Named("condition") = r_condition,
            Named("difficulty") = r_difficulty,
            Named("info") = r_info
        );
    }
    
    return List::create(
        Named("colnames") = r_colnames,
        Named("subjects") = r_subjects
    );
}