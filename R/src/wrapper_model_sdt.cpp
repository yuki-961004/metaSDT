#include <Rcpp.h>
#include <model_sdt.hpp>

// 使用宏定义解决多重编译问题
#define CORE_IMPL "Cpp/src/model_sdt.cpp"
#include CORE_IMPL

using namespace Rcpp;

//' Evaluate SDT Model CDFs
// [[Rcpp::export]]
List model_sdt(List params, NumericVector x) {
    // 1. 将 R 侧已经拍扁的 List 转换为 C++ 的字典
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    if (params.size() > 0 && params.hasAttribute("names")) {
        CharacterVector names = params.names();
        for (int i = 0; i < params.size(); ++i) {
            cpp_params[as<std::string>(names[i])] = as<std::vector<double>>(params[i]);
        }
    }

    // 2. 初始化数学模型
    ModelSDT model(cpp_params);

    // 3. 将 R 向量转为 C++ 向量并进行批量计算
    std::vector<double> cpp_x = as<std::vector<double>>(x);

    // 4. 返回包含两组 CDF 结果的列表
    return List::create(
        Named("cdf_noise") = model.cdf_noise(cpp_x),
        Named("cdf_signal") = model.cdf_signal(cpp_x)
    );
}