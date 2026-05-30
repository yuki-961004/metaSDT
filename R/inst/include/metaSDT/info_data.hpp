#ifndef INFO_DATA_HPP
#define INFO_DATA_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <map>

// 定义单个被试的统计信息
struct SubjectInfo {
    int n_trials = 0;
    int n_blocks = 0;
};

// 定义单个被试的数据视图(View)
struct DataInfoSubject {
    // 轻量级引用: 该被试在原数据集中对应的所有行索引
    std::vector<int> raw; 
    
    std::unordered_map<std::string, std::vector<int>> condition; 
    
    // 按照 difficulty (难度) 切分好的行索引
    std::unordered_map<std::string, std::vector<int>> difficulty;

    // 试次和 block 的统计信息
    SubjectInfo info; 
};

// 定义整个函数的全局返回结果
struct DataInfoResult {
    std::unordered_map<std::string, std::string> colnames; // 记录正则匹配最终敲定的列名
    std::map<double, DataInfoSubject> subjects;            // 存储按被试 ID 拆分的视图结构(使用 map 保证 ID 排序)
    std::vector<std::string> messages;                     // 存储扫描过程中产生的非致命提示信息
};

DataInfoResult info_data(
    const std::unordered_map<std::string, std::vector<double>>& df, // 假设传入的 DataFrame 被拍扁为双精度浮点列字典
    const std::unordered_map<std::string, std::string>& colnames = {}
);

#endif // INFO_DATA_HPP
