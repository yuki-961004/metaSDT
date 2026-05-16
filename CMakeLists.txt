cmake_minimum_required(VERSION 3.15...3.27)
project(metaSDT_Core CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # 如果没有指定安装路径，默认安装到构建目录下的 install 文件夹中
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Default install prefix" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# [关键设置] 关闭编译器特有扩展 (如 gnu++17), 保证底层数学库在全平台的绝对兼容性.
set(CMAKE_CXX_EXTENSIONS OFF)

# 全局拦截并禁用对 Java 和 JNI 的查找，消除第三方库产生的无关警告
set(CMAKE_DISABLE_FIND_PACKAGE_Java ON)
set(CMAKE_DISABLE_FIND_PACKAGE_JNI ON)

# 1. 启用 OpenMP 支持多线程并行加速
find_package(OpenMP REQUIRED)
if(MSVC)
    add_compile_options(/utf-8 /wd4819 /wd4267 /wd4101) # 强制 UTF-8 编码并屏蔽 MSVC 警告
endif()

# 2. 现场拉取并极简编译 NLopt
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  nlopt
  GIT_REPOSITORY https://github.com/stevengj/nlopt.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)
# 强制关闭不需要的组件，只编译纯净的静态优化库
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(NLOPT_SHARED OFF CACHE BOOL "" FORCE)
set(NLOPT_TESTS OFF CACHE BOOL "" FORCE)
set(NLOPT_CXX ON CACHE BOOL "" FORCE)
set(NLOPT_CXX11 OFF CACHE BOOL "" FORCE)
set(NLOPT_FORTRAN OFF CACHE BOOL "" FORCE)
set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE)
set(NLOPT_GUILE OFF CACHE BOOL "" FORCE)
set(NLOPT_MATLAB OFF CACHE BOOL "" FORCE)
set(NLOPT_OCTAVE OFF CACHE BOOL "" FORCE)
set(NLOPT_SWIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlopt)

# Stan Math is pulled in as a header-oriented dependency for estimate_mcmc.
# The sampler uses Stan Math finite-difference gradients around the existing
# CriterionPosterior engine, so no generated Stan model file is needed here.
set(METASDT_STAN_MATH_TAG "v5.2.0" CACHE STRING "Stan Math tag to fetch")
FetchContent_Declare(
  stan_math
  GIT_REPOSITORY https://github.com/stan-dev/math.git
  GIT_TAG ${METASDT_STAN_MATH_TAG}
  GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(stan_math)

file(GLOB METASDT_STAN_EIGEN_DIRS
     CONFIGURE_DEPENDS "${stan_math_SOURCE_DIR}/lib/eigen_*")
file(GLOB METASDT_STAN_BOOST_DIRS
     CONFIGURE_DEPENDS "${stan_math_SOURCE_DIR}/lib/boost_*")
file(GLOB METASDT_STAN_SUNDIALS_DIRS
     CONFIGURE_DEPENDS "${stan_math_SOURCE_DIR}/lib/sundials_*")
file(GLOB METASDT_STAN_TBB_DIRS
     CONFIGURE_DEPENDS "${stan_math_SOURCE_DIR}/lib/tbb_*")

set(METASDT_STAN_MATH_INCLUDE_DIRS ${stan_math_SOURCE_DIR})
list(APPEND METASDT_STAN_MATH_INCLUDE_DIRS ${METASDT_STAN_EIGEN_DIRS})
list(APPEND METASDT_STAN_MATH_INCLUDE_DIRS ${METASDT_STAN_BOOST_DIRS})
foreach(stan_sundials_dir ${METASDT_STAN_SUNDIALS_DIRS})
    list(APPEND METASDT_STAN_MATH_INCLUDE_DIRS "${stan_sundials_dir}/include")
endforeach()
foreach(stan_tbb_dir ${METASDT_STAN_TBB_DIRS})
    list(APPEND METASDT_STAN_MATH_INCLUDE_DIRS "${stan_tbb_dir}/include")
endforeach()

# 3. 编译核心 C++ 源码为单独的静态库 (注意现在的路径以项目根目录为基准)
add_library(metaSDT_core STATIC
    Cpp/src/modify_params.cpp
    Cpp/src/modify_prior.cpp
    Cpp/src/modify_control.cpp
    Cpp/src/modify_outputs.cpp

    Cpp/src/info_data.cpp
    Cpp/src/info_nlopt.cpp

    Cpp/src/matrix_freq.cpp
    Cpp/src/matrix_prob.cpp
    Cpp/src/matrix_mult.cpp

    Cpp/src/model_sdt.cpp

    Cpp/src/algorithm_nlopt.cpp
    Cpp/src/algorithm_stan.cpp
    Cpp/src/task_builder.cpp
    Cpp/src/estimate_mle.cpp
    Cpp/src/estimate_map.cpp
    Cpp/src/estimate_mcmc.cpp

    Cpp/src/progress_bar.cpp
)

# 开启 -fPIC 确保能够链接入后续的 Python/R 动态库中
set_property(TARGET metaSDT_core PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(metaSDT_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cpp/include>
    $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
    $<BUILD_INTERFACE:${nlopt_BINARY_DIR}>
    $<BUILD_INTERFACE:${nlopt_SOURCE_DIR}/src/api>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(
    metaSDT_core
    PRIVATE
    ${METASDT_STAN_MATH_INCLUDE_DIRS}
)

# [解耦修复] Eigen 现在作为纯本地头文件存在, 不再作为 CMake Target 链接
# Stan Math 仅供 estimate_mcmc 内部编译使用, 不暴露到公共头文件接口
target_link_libraries(metaSDT_core PUBLIC nlopt OpenMP::OpenMP_CXX)
target_link_libraries(metaSDT_core PUBLIC nlopt OpenMP::OpenMP_CXX)

# ==============================================================================
# 4. 导出与安装系统 (Export & Install) - 预编译解耦的核心
# ==============================================================================
include(GNUInstallDirs)

# 4.1 安装前清理旧产物 (仅清理 metaSDT 自己的安装内容)
# 目的: 保留增量编译速度, 同时避免 install 前缀中残留过时文件
option(METASDT_PREINSTALL_CLEAN
       "Remove stale installed metaSDT artifacts before install"
       ON)

if(METASDT_PREINSTALL_CLEAN)
    install(CODE "
        message(STATUS \"metaSDT: cleaning stale installed artifacts...\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -rf
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/Eigen\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -rf
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/metaSDT\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -rf
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/metaSDT_core\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -f
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libmetaSDT_core.a\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -f
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/metaSDT_core.lib\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -f
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/metaSDT_core.dll\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -f
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/mcmc_control.hpp\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -f
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/mcmc_result.hpp\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -f
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/sampler_hmc.hpp\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -f
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/build_objective.hpp\")
    ")
endif()

# 导出目标 (静态库本体)
install(TARGETS metaSDT_core EXPORT metaSDT_Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# 导出头文件
install(DIRECTORY Cpp/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# [核心修复] 连带 Eigen 的头文件一起导出, 防止 Python 包装器编译时报错找不到 <Eigen/Dense>
install(DIRECTORY ${eigen_SOURCE_DIR}/Eigen DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# 导出供 find_package 使用的 Target 配置
install(EXPORT metaSDT_Targets 
        FILE metaSDT_coreTargets.cmake 
        NAMESPACE metaSDT::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metaSDT_core)

# 自动生成 Config.cmake (让外部项目能 find_package(metaSDT_core))
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/metaSDT_coreConfig.cmake"
     "include(CMakeFindDependencyMacro)\n"
     "find_dependency(OpenMP)\n"
     "find_dependency(NLopt)\n"
     "include(\"\${CMAKE_CURRENT_LIST_DIR}/metaSDT_coreTargets.cmake\")\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/metaSDT_coreConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metaSDT_core)

# 5. 根据构建环境挂载前端拓展 (保留后门兼容全量构建)
if(SKBUILD) # SKBUILD 是 scikit-build-core 启动时自动注入的标志
    add_subdirectory(Python)
endif()
