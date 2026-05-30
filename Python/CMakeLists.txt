cmake_minimum_required(VERSION 3.15...3.27)
project(metaSDT_Python_Wrapper CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_FIND_PACKAGE_Java ON)
set(CMAKE_DISABLE_FIND_PACKAGE_JNI ON)

option(
    METASDT_FETCH_NLOPT
    "Fetch NLopt instead of using a system installation"
    ON
)

option(
    METASDT_ENABLE_MCMC
    "Build MCMC estimator with Stan Math support"
    ON
)

set(
    METASDT_STAN_MATH_TAG
    "v5.2.0"
    CACHE STRING
    "Stan Math tag to fetch"
)

if(MSVC)
    add_compile_options(/utf-8 /wd4819 /wd4267)
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

set(METASDT_EIGEN_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/eigen-src")
set(METASDT_EIGEN_BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/eigen-build")
set(METASDT_EIGEN_SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/eigen-subbuild")

if(NOT EXISTS "${METASDT_EIGEN_SOURCE_DIR}/Eigen")
    FetchContent_Populate(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        SOURCE_DIR "${METASDT_EIGEN_SOURCE_DIR}"
        BINARY_DIR "${METASDT_EIGEN_BINARY_DIR}"
        SUBBUILD_DIR "${METASDT_EIGEN_SUBBUILD_DIR}"
    )
else()
    set(eigen_SOURCE_DIR "${METASDT_EIGEN_SOURCE_DIR}")
endif()

FetchContent_Declare(
    abcpp
    GIT_REPOSITORY https://github.com/yuki-961004/abcpp.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(abcpp)

set(METASDT_NLOPT_TARGET "")
set(METASDT_NLOPT_INCLUDE_DIRS "")

if(METASDT_FETCH_NLOPT)
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

    FetchContent_Declare(
        nlopt
        GIT_REPOSITORY https://github.com/stevengj/nlopt.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlopt)

    set(METASDT_NLOPT_TARGET nlopt)
    list(APPEND METASDT_NLOPT_INCLUDE_DIRS "${nlopt_BINARY_DIR}")
    list(APPEND METASDT_NLOPT_INCLUDE_DIRS "${nlopt_SOURCE_DIR}/src/api")
else()
    find_package(NLopt REQUIRED)

    if(TARGET NLopt::nlopt)
        set(METASDT_NLOPT_TARGET NLopt::nlopt)
    elseif(TARGET nlopt)
        set(METASDT_NLOPT_TARGET nlopt)
    else()
        message(FATAL_ERROR "NLopt has no usable CMake target.")
    endif()
endif()

set(METASDT_STAN_MATH_INCLUDE_DIRS "")

if(METASDT_ENABLE_MCMC)
    FetchContent_Declare(
        stan_math
        GIT_REPOSITORY https://github.com/stan-dev/math.git
        GIT_TAG ${METASDT_STAN_MATH_TAG}
        GIT_SHALLOW TRUE
        GIT_SUBMODULES_RECURSE TRUE
    )
    FetchContent_MakeAvailable(stan_math)

    file(
        GLOB METASDT_STAN_EIGEN_DIRS
        CONFIGURE_DEPENDS
        "${stan_math_SOURCE_DIR}/lib/eigen_*"
    )
    file(
        GLOB METASDT_STAN_BOOST_DIRS
        CONFIGURE_DEPENDS
        "${stan_math_SOURCE_DIR}/lib/boost_*"
    )
    file(
        GLOB METASDT_STAN_SUNDIALS_DIRS
        CONFIGURE_DEPENDS
        "${stan_math_SOURCE_DIR}/lib/sundials_*"
    )
    file(
        GLOB METASDT_STAN_TBB_DIRS
        CONFIGURE_DEPENDS
        "${stan_math_SOURCE_DIR}/lib/tbb_*"
    )

    list(APPEND METASDT_STAN_MATH_INCLUDE_DIRS "${stan_math_SOURCE_DIR}")
    list(APPEND METASDT_STAN_MATH_INCLUDE_DIRS ${METASDT_STAN_EIGEN_DIRS})
    list(APPEND METASDT_STAN_MATH_INCLUDE_DIRS ${METASDT_STAN_BOOST_DIRS})

    foreach(stan_sundials_dir ${METASDT_STAN_SUNDIALS_DIRS})
        list(
            APPEND METASDT_STAN_MATH_INCLUDE_DIRS
            "${stan_sundials_dir}/include"
        )
    endforeach()

    foreach(stan_tbb_dir ${METASDT_STAN_TBB_DIRS})
        list(
            APPEND METASDT_STAN_MATH_INCLUDE_DIRS
            "${stan_tbb_dir}/include"
        )
    endforeach()
endif()

set(
    METASDT_PY_BACKEND_SOURCES
    src/cpp/modify_params.cpp
    src/cpp/modify_priors.cpp
    src/cpp/modify_control.cpp
    src/cpp/modify_outputs.cpp
    src/cpp/info_data.cpp
    src/cpp/info_nlopt.cpp
    src/cpp/matrix_freq.cpp
    src/cpp/matrix_prob.cpp
    src/cpp/matrix_mult.cpp
    src/cpp/model_sdt.cpp
    src/cpp/shell_run_m.cpp
    src/cpp/algorithm_nlopt.cpp
    src/cpp/algorithm_abcpp.cpp
    src/cpp/task_builder.cpp
    src/cpp/estimate_mle.cpp
    src/cpp/estimate_map.cpp
    src/cpp/estimate_abc.cpp
    src/cpp/progress_bar.cpp
)

if(METASDT_ENABLE_MCMC)
    list(
        APPEND METASDT_PY_BACKEND_SOURCES
        src/cpp/algorithm_stan.cpp
        src/cpp/estimate_mcmc.cpp
    )
endif()

add_library(metaSDT_python_core STATIC ${METASDT_PY_BACKEND_SOURCES})
set_property(TARGET metaSDT_python_core PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(
    metaSDT_python_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
        "${eigen_SOURCE_DIR}"
)

foreach(nlopt_include_dir ${METASDT_NLOPT_INCLUDE_DIRS})
    target_include_directories(
        metaSDT_python_core
        PUBLIC
            "${nlopt_include_dir}"
    )
endforeach()

if(METASDT_ENABLE_MCMC)
    target_include_directories(
        metaSDT_python_core
        PRIVATE
            ${METASDT_STAN_MATH_INCLUDE_DIRS}
    )

    target_compile_definitions(
        metaSDT_python_core
        PUBLIC
            METASDT_ENABLE_MCMC
    )
endif()

target_link_libraries(
    metaSDT_python_core
    PUBLIC
        ${METASDT_NLOPT_TARGET}
        abcpp::abcpp
        OpenMP::OpenMP_CXX
)

macro(add_meta_module module_name source_file)
    pybind11_add_module(${module_name} MODULE ${source_file})
    target_link_libraries(${module_name} PRIVATE metaSDT_python_core)
    target_include_directories(
        ${module_name}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
    )
    install(TARGETS ${module_name} DESTINATION metaSDT)
endmacro()

add_meta_module(_core_matrix_freq src/py_matrix_freq.cpp)
add_meta_module(_core_matrix_prob src/py_matrix_prob.cpp)
add_meta_module(_core_matrix_mult src/py_matrix_mult.cpp)
add_meta_module(_core_criterion_likelihood src/py_criterion_likelihood.cpp)
add_meta_module(_core_criterion_prior src/py_criterion_prior.cpp)
add_meta_module(_core_criterion_posterior src/py_criterion_posterior.cpp)
add_meta_module(_help_modify_params src/py_modify_params.cpp)
add_meta_module(_help_modify_priors src/py_modify_priors.cpp)
add_meta_module(_help_info_data src/py_info_data.cpp)
add_meta_module(_model_sdt src/py_model_sdt.cpp)
add_meta_module(_shell_run_m src/py_shell_run_m.cpp)
add_meta_module(_estimate_mle src/py_estimate_mle.cpp)
add_meta_module(_estimate_map src/py_estimate_map.cpp)
if(METASDT_ENABLE_MCMC)
    add_meta_module(_estimate_mcmc src/py_estimate_mcmc.cpp)
endif()
add_meta_module(_estimate_abc src/py_estimate_abc.cpp)
add_meta_module(_progress_bar src/py_progress_bar.cpp)
