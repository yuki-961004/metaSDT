cmake_minimum_required(VERSION 3.15...3.27)
project(metaSDT_Python_Wrapper CXX)

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_compile_options(/utf-8 /wd4819 /wd4267)
endif()

get_filename_component(
    METASDT_REPO_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ABSOLUTE
)
set(
    METASDT_CORE_INSTALL_PREFIX
    "${METASDT_REPO_ROOT}/build/install"
    CACHE PATH
    "Install prefix that contains metaSDT_coreConfig.cmake"
)
if(EXISTS "${METASDT_CORE_INSTALL_PREFIX}")
    list(PREPEND CMAKE_PREFIX_PATH "${METASDT_CORE_INSTALL_PREFIX}")
endif()

find_package(pybind11 CONFIG REQUIRED)

if(NOT TARGET metaSDT_core)
    find_package(metaSDT_core REQUIRED)
endif()

macro(add_meta_module module_name source_file)
    pybind11_add_module(${module_name} MODULE ${source_file})
    if(TARGET metaSDT::metaSDT_core)
        target_link_libraries(${module_name} PRIVATE metaSDT::metaSDT_core)
    else()
        target_link_libraries(${module_name} PRIVATE metaSDT_core)
    endif()
    install(TARGETS ${module_name} DESTINATION metaSDT)
endmacro()

add_meta_module(_core_matrix_freq src/py_matrix_freq.cpp)
add_meta_module(_core_matrix_prob src/py_matrix_prob.cpp)
add_meta_module(_core_matrix_mult src/py_matrix_mult.cpp)
add_meta_module(_core_criterion_likelihood src/py_criterion_likelihood.cpp)
add_meta_module(_core_criterion_prior src/py_criterion_prior.cpp)
add_meta_module(_core_criterion_posterior src/py_criterion_posterior.cpp)
add_meta_module(_help_modify_params src/py_modify_params.cpp)
add_meta_module(_help_modify_prior src/py_modify_prior.cpp)
add_meta_module(_help_info_data src/py_info_data.cpp)
add_meta_module(_model_sdt src/py_model_sdt.cpp)
add_meta_module(_estimate_mle src/py_estimate_mle.cpp)
add_meta_module(_estimate_map src/py_estimate_map.cpp)
add_meta_module(_estimate_mcmc src/py_estimate_mcmc.cpp)
add_meta_module(_progress_bar src/py_progress_bar.cpp)
