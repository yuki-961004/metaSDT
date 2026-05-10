# 1. 引入 pybind11
find_package(pybind11 CONFIG REQUIRED)

# 2. 注册宏 (模块只依赖顶层的 metaSDT_core，无需关心 nlopt 等细节)
macro(add_meta_module module_name source_file)
    pybind11_add_module(${module_name} MODULE ${source_file})
    target_link_libraries(${module_name} PRIVATE metaSDT_core)
    # 告诉构建系统，编译好的动态库放在 metaSDT/ 目录下
    install(TARGETS ${module_name} DESTINATION metaSDT)
endmacro()

# 3. 逐个挂载 Python Wrappers
add_meta_module(_core_matrix_freq src/wrapper_matrix_freq.cpp)
add_meta_module(_core_matrix_prob src/wrapper_matrix_prob.cpp)
add_meta_module(_core_matrix_mult src/wrapper_matrix_mult.cpp)
add_meta_module(_core_loss_function src/wrapper_loss_function.cpp)
add_meta_module(_help_modify_params src/wrapper_modify_params.cpp)
add_meta_module(_core_data_info src/wrapper_data_info.cpp)
add_meta_module(_model_sdt src/wrapper_model_sdt.cpp)
add_meta_module(_estimate_mle src/wrapper_estimate_mle.cpp)
