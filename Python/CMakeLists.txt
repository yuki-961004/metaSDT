cmake_minimum_required(VERSION 3.15...3.27)
project(metaSDT_Python_Wrapper CXX)

set(CMAKE_CXX_STANDARD 17)

get_filename_component(METASDT_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(METASDT_CORE_INSTALL_PREFIX "${METASDT_REPO_ROOT}/Cpp/build/install" CACHE PATH
    "Install prefix that contains metaSDT_coreConfig.cmake")
if(EXISTS "${METASDT_CORE_INSTALL_PREFIX}")
    list(PREPEND CMAKE_PREFIX_PATH "${METASDT_CORE_INSTALL_PREFIX}")
endif()

# 1. 引入 pybind11
find_package(pybind11 CONFIG REQUIRED)

# 2. 寻找预先编译好的 metaSDT 核心库
if(NOT TARGET metaSDT_core)
    find_package(metaSDT_core REQUIRED)
endif()

# 3. 注册宏 (兼容独立编译与一体化编译)
macro(add_meta_module module_name source_file)
    pybind11_add_module(${module_name} MODULE ${source_file})
    if(TARGET metaSDT::metaSDT_core)
        target_link_libraries(${module_name} PRIVATE metaSDT::metaSDT_core)
    else()
        target_link_libraries(${module_name} PRIVATE metaSDT_core)
    endif()
    # 告诉构建系统，编译好的动态库放在 metaSDT/ 目录下
    install(TARGETS ${module_name} DESTINATION metaSDT)
endmacro()

# 4. 逐个挂载 Python Wrappers
add_meta_module(_core_matrix_freq src/wrapper_matrix_freq.cpp)
add_meta_module(_core_matrix_prob src/wrapper_matrix_prob.cpp)
add_meta_module(_core_matrix_mult src/wrapper_matrix_mult.cpp)
add_meta_module(_core_criterion_likelihood src/wrapper_criterion_likelihood.cpp)
add_meta_module(_core_criterion_prior src/wrapper_criterion_prior.cpp)
add_meta_module(_core_criterion_posterior src/wrapper_criterion_posterior.cpp)
add_meta_module(_help_modify_params src/wrapper_modify_params.cpp)
add_meta_module(_help_modify_prior src/wrapper_modify_prior.cpp)
add_meta_module(_help_data_info src/wrapper_data_info.cpp)
add_meta_module(_model_sdt src/wrapper_model_sdt.cpp)
add_meta_module(_estimate_mle src/wrapper_estimate_mle.cpp)
