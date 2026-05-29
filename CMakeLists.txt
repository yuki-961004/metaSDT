cmake_minimum_required(VERSION 3.15...3.27)

############################
# Project setup
############################

project(metaSDT CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(
        CMAKE_INSTALL_PREFIX
        "${CMAKE_BINARY_DIR}/install"
        CACHE PATH
        "Default install prefix"
        FORCE
    )
endif()

include(FetchContent)
include(GNUInstallDirs)

############################
# Compiler settings
############################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_FIND_PACKAGE_Java ON)
set(CMAKE_DISABLE_FIND_PACKAGE_JNI ON)

find_package(OpenMP REQUIRED)

if(MSVC)
    add_compile_options(/utf-8 /wd4819 /wd4267 /wd4101)
endif()

############################
# Options
############################

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

option(
    METASDT_PREINSTALL_CLEAN
    "Remove stale installed metaSDT artifacts before install"
    ON
)

set(
    METASDT_STAN_MATH_TAG
    "v5.2.0"
    CACHE STRING
    "Stan Math tag to fetch"
)

############################
# Dependencies
############################

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

set(METASDT_PARENT_SKBUILD ${SKBUILD})
set(SKBUILD OFF)
FetchContent_MakeAvailable(abcpp)
set(SKBUILD ${METASDT_PARENT_SKBUILD})

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

############################
# Source files
############################

set(
    METASDT_SOURCES
    Cpp/src/modify_params.cpp
    Cpp/src/modify_priors.cpp
    Cpp/src/modify_control.cpp
    Cpp/src/modify_outputs.cpp
    Cpp/src/info_data.cpp
    Cpp/src/info_nlopt.cpp
    Cpp/src/matrix_freq.cpp
    Cpp/src/matrix_prob.cpp
    Cpp/src/matrix_mult.cpp
    Cpp/src/model_sdt.cpp
    Cpp/src/algorithm_nlopt.cpp
    Cpp/src/algorithm_abcpp.cpp
    Cpp/src/task_builder.cpp
    Cpp/src/estimate_mle.cpp
    Cpp/src/estimate_map.cpp
    Cpp/src/estimate_abc.cpp
    Cpp/src/progress_bar.cpp
)

if(METASDT_ENABLE_MCMC)
    list(
        APPEND METASDT_SOURCES
        Cpp/src/algorithm_stan.cpp
        Cpp/src/estimate_mcmc.cpp
    )
endif()

############################
# Target definition
############################

add_library(metaSDT STATIC ${METASDT_SOURCES})

set_property(TARGET metaSDT PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(
    metaSDT
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cpp/include>
        $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

foreach(nlopt_include_dir ${METASDT_NLOPT_INCLUDE_DIRS})
    target_include_directories(
        metaSDT
        PUBLIC
            $<BUILD_INTERFACE:${nlopt_include_dir}>
    )
endforeach()

if(METASDT_ENABLE_MCMC)
    target_include_directories(
        metaSDT
        PRIVATE
            ${METASDT_STAN_MATH_INCLUDE_DIRS}
    )

    target_compile_definitions(metaSDT PUBLIC METASDT_ENABLE_MCMC)
endif()

target_link_libraries(
    metaSDT
    PUBLIC
        ${METASDT_NLOPT_TARGET}
        abcpp::abcpp
        OpenMP::OpenMP_CXX
)

############################
# Install/export
############################

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
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/metaSDT\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -f
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libmetaSDT.a\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -f
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/metaSDT.lib\")
        execute_process(
            COMMAND \"\${CMAKE_COMMAND}\" -E rm -f
                    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/metaSDT.dll\")
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

install(
    TARGETS metaSDT
    EXPORT metaSDT_Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY Cpp/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${eigen_SOURCE_DIR}/Eigen DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
    EXPORT metaSDT_Targets
    FILE metaSDTTargets.cmake
    NAMESPACE metaSDT::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metaSDT
)

file(
    WRITE "${CMAKE_CURRENT_BINARY_DIR}/metaSDTConfig.cmake"
    "include(CMakeFindDependencyMacro)\n"
    "find_dependency(OpenMP)\n"
    "find_dependency(NLopt)\n"
    "find_dependency(abcpp)\n"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/metaSDTTargets.cmake\")\n"
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/metaSDTConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metaSDT
)

############################
# Frontend integration
############################

if(SKBUILD)
    add_subdirectory(Python)
endif()
