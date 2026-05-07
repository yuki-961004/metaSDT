#ifndef DATA_INFO_HPP
#define DATA_INFO_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <map>

// 定义单个被试的统计信息
struct SubjectInfo {
    int n_trials = 0;
    int n_blocks = 0;
};

// 定义单个被试的数据视图 (View)
struct DataInfoSubject {
    // 轻量级引用：该被试在原数据集中对应的所有行索引
    std::vector<int> raw; 
    
    // 按照 condition 切分好的行索引 (键为条件拼接的字符串，值为行索引数组)
    // 这完美实现了你不占用额外内存、“只划分界限”的设计！
    std::unordered_map<std::string, std::vector<int>> condition; 
    
    // 试次与 block 的统计信息
    SubjectInfo info; 
};

// 定义整个函数的全局返回结果
struct DataInfoResult {
    std::unordered_map<std::string, std::string> colnames; // 记录正则匹配最终敲定的列名
    std::map<double, DataInfoSubject> subjects;            // 存储按被试 ID 拆分的视图结果 (使用 map 保证按 ID 排序)
    std::vector<std::string> warnings;                     // 存储扫描过程中产生的非致命警告
};

DataInfoResult data_info_core(
    const std::unordered_map<std::string, std::vector<double>>& df, // 假设传入的 DataFrame 被拍扁为双精度浮点列字典
    const std::unordered_map<std::string, std::string>& colnames,
    const std::vector<std::string>& condition
);

#endif // DATA_INFO_HPP