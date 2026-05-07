#include <Rcpp.h>
#include "Cpp/include/model_sdt.hpp"

// 使用宏定义包住 include，骗过 Rcpp::sourceCpp 的正则检查
#define CORE_IMPL "Cpp/src/model_sdt.cpp"
#include CORE_IMPL

using namespace Rcpp;

// [[Rcpp::export]]
List model_sdt(List params) { // <- 注意：这里不再需要传入坐标了，只要参数字典！
    std::unordered_map<std::string, std::vector<double>> cpp_params;
    if (params.size() > 0 && params.hasAttribute("names")) {
        CharacterVector names = params.names();
        for (int i = 0; i < params.size(); ++i) {
            std::string key = as<std::string>(names[i]);
            // 拦截包含元数据的槽位，防止将其强制转换为 double 数组时报错
            if (key == "free_params" || key == "fixed_params" || key == "constant_params") continue;
            cpp_params[key] = as<std::vector<double>>(params[i]);
        }
    }
    ModelSDT model(cpp_params);
    
    return List::create(Named("cdf_noise") = model.cdf_noise(),
                        Named("cdf_signal") = model.cdf_signal());
}