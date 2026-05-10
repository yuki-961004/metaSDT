cmake_minimum_required(VERSION 3.15...3.27)
project(metaSDT_Python)

# 1. 引入 pybind11 (scikit-build-core 会自动配置 Python 环境)
find_package(pybind11 CONFIG REQUIRED)

# 启用 OpenMP 支持底层多线程并行加速
find_package(OpenMP REQUIRED)
if(MSVC)
    add_compile_options(/wd4819 /wd4267 /wd4101) # 屏蔽 MSVC 字符集和类型截断警告
endif()

# 2. 现场拉取并极简编译 NLopt
include(FetchContent)
FetchContent_Declare(
  nlopt
  GIT_REPOSITORY https://github.com/stevengj/nlopt.git
  GIT_TAG master
)
# 强制关闭所有不需要的组件（避免污染 Python 环境和触发其他环境依赖报错）
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(NLOPT_SHARED OFF CACHE BOOL "" FORCE)
set(NLOPT_TESTS OFF CACHE BOOL "" FORCE)
set(NLOPT_CXX OFF CACHE BOOL "" FORCE)
set(NLOPT_CXX11 OFF CACHE BOOL "" FORCE)
set(NLOPT_FORTRAN OFF CACHE BOOL "" FORCE)
set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE) # 绝不编译 NLopt 自带的 Python Wrapper
set(NLOPT_GUILE OFF CACHE BOOL "" FORCE)
set(NLOPT_MATLAB OFF CACHE BOOL "" FORCE)
set(NLOPT_OCTAVE OFF CACHE BOOL "" FORCE)
set(NLOPT_SWIG OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlopt)

# 3. 定义一个宏，快速注册 pybind11 模块并设置输出路径
macro(add_meta_module module_name source_file)
    pybind11_add_module(${module_name} MODULE ${source_file})
    # 告诉构建系统，编译好的动态库放在 metaSDT/ 目录下
    install(TARGETS ${module_name} DESTINATION metaSDT)
endmacro()

# 4. 逐个注册我们所有的 C++ Wrappers
add_meta_module(_core_matrix_freq src/wrapper_matrix_freq.cpp)
add_meta_module(_core_matrix_prob src/wrapper_matrix_prob.cpp)
add_meta_module(_core_matrix_mult src/wrapper_matrix_mult.cpp)
add_meta_module(_core_loss_function src/wrapper_loss_function.cpp)
add_meta_module(_help_modify_params src/wrapper_modify_params.cpp)
add_meta_module(_core_data_info src/wrapper_data_info.cpp)
add_meta_module(_model_sdt src/wrapper_model_sdt.cpp)

# 5. 单独处理 _estimate_mle，因为它需要链接我们刚现场编译的 nlopt 静态库！
add_meta_module(_estimate_mle src/wrapper_estimate_mle.cpp)
target_link_libraries(_estimate_mle PRIVATE nlopt OpenMP::OpenMP_CXX)
