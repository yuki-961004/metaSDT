cmake_minimum_required(VERSION 3.15...3.27)
project(metaSDT_Core CXX)

# 1. 启用 OpenMP 支持多线程并行加速
find_package(OpenMP REQUIRED)
if(MSVC)
    add_compile_options(/wd4819 /wd4267 /wd4101) # 屏蔽 MSVC 警告
endif()

# 2. 现场拉取并极简编译 NLopt
include(FetchContent)
FetchContent_Declare(
  nlopt
  GIT_REPOSITORY https://github.com/stevengj/nlopt.git
  GIT_TAG master
)
# 强制关闭不需要的组件，只编译纯净的静态优化库
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(NLOPT_SHARED OFF CACHE BOOL "" FORCE)
set(NLOPT_TESTS OFF CACHE BOOL "" FORCE)
set(NLOPT_CXX OFF CACHE BOOL "" FORCE)
set(NLOPT_CXX11 OFF CACHE BOOL "" FORCE)
set(NLOPT_FORTRAN OFF CACHE BOOL "" FORCE)
set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE)
set(NLOPT_GUILE OFF CACHE BOOL "" FORCE)
set(NLOPT_MATLAB OFF CACHE BOOL "" FORCE)
set(NLOPT_OCTAVE OFF CACHE BOOL "" FORCE)
set(NLOPT_SWIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlopt)

# 2.5 现场拉取 Eigen 库 (Header-only, 编译极快)
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    # URL_HASH SHA256=... (optional but good practice)
)
FetchContent_MakeAvailable(eigen)

# 3. 编译核心 C++ 源码为单独的静态库 (注意现在的路径以项目根目录为基准)
add_library(metaSDT_core STATIC
    Cpp/src/modify_params.cpp
    Cpp/src/modify_prior.cpp
    Cpp/src/data_info.cpp
    Cpp/src/matrix_freq.cpp
    Cpp/src/model_sdt.cpp
    Cpp/src/matrix_prob.cpp
    Cpp/src/matrix_mult.cpp
    Cpp/src/objective_function.cpp
    Cpp/src/estimate_mle.cpp
)

# 开启 fPIC 确保能够链接入后续的 Python/R 动态库中
set_property(TARGET metaSDT_core PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(metaSDT_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cpp/include>)
target_link_libraries(metaSDT_core PUBLIC nlopt OpenMP::OpenMP_CXX Eigen3::Eigen)

# 4. 根据构建环境挂载前端拓展
if(SKBUILD) # SKBUILD 是 scikit-build-core 启动时自动注入的标志
    add_subdirectory(Python)
endif()