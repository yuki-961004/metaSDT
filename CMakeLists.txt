cmake_minimum_required(VERSION 3.15...3.27)
project(metaSDT_Core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# [鍏抽敭璁剧疆] 鍏抽棴缂栬瘧鍣ㄧ壒鏈夋墿灞?濡?gnu++17), 淇濊瘉搴曞眰鏁板搴撳湪鍏ㄥ钩鍙扮殑缁濆鍏煎鎬?
set(CMAKE_CXX_EXTENSIONS OFF)

# 1. 鍚敤 OpenMP 鏀寔澶氱嚎绋嬪苟琛屽姞閫?
find_package(OpenMP REQUIRED)
if(MSVC)
    add_compile_options(/utf-8 /wd4819 /wd4267 /wd4101) # 寮哄埗 UTF-8 缂栫爜骞跺睆钄?MSVC 璀﹀憡
endif()

# 2. 鐜板満鎷夊彇骞舵瀬绠€缂栬瘧 NLopt
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
# 寮哄埗鍏抽棴涓嶉渶瑕佺殑缁勪欢锛屽彧缂栬瘧绾噣鐨勯潤鎬佷紭鍖栧簱
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
FetchContent_GetProperties(stan_math)
if(NOT stan_math_POPULATED)
  FetchContent_Populate(stan_math)
endif()

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

# 3. 缂栬瘧鏍稿績 C++ 婧愮爜涓哄崟鐙殑闈欐€佸簱 (娉ㄦ剰鐜板湪鐨勮矾寰勪互椤圭洰鏍圭洰褰曚负鍩哄噯)
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

# 寮€鍚?fPIC 纭繚鑳藉閾炬帴鍏ュ悗缁殑 Python/R 鍔ㄦ€佸簱涓?
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

# [瑙ｈ€︿慨澶峕 Eigen 鐜板湪浣滀负绾湰鍦板ご鏂囦欢瀛樺湪, 涓嶅啀浣滀负 CMake Target 閾炬帴
# Stan Math 浠呬緵 estimate_mcmc 鍐呴儴缂栬瘧浣跨敤, 涓嶆毚闇插埌鍏叡澶存枃浠舵帴鍙?target_link_libraries(metaSDT_core PUBLIC nlopt OpenMP::OpenMP_CXX)

# ==============================================================================
# 4. 瀵煎嚭涓庡畨瑁呯郴?(Export & Install) - 棰勭紪璇戣В鑰︾殑鏍稿績
# ==============================================================================
include(GNUInstallDirs)

# 4.1 瀹夎鍓嶆竻鐞嗘棫浜х墿(浠呮竻鐞?metaSDT 鑷繁鐨勫畨瑁呭唴瀹?
# 鐩殑: 淇濈暀澧為噺缂栬瘧閫熷害, 鍚屾椂閬垮厤 install 鍓嶇紑涓畫鐣欒繃鏃舵枃浠?
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

# 瀵煎嚭鐩爣?(闈欐€佸簱鏈綋)
install(TARGETS metaSDT_core EXPORT metaSDT_Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# 瀵煎嚭澶存枃?
install(DIRECTORY Cpp/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# [鏍稿績淇] 杩炲甫 Eigen 鐨勫ご鏂囦欢涓€璧峰鍑? 闃叉 Python 鍖呰鍣ㄧ紪璇戞椂鎶ユ壘涓嶅埌 <Eigen/Dense>
install(DIRECTORY ${eigen_SOURCE_DIR}/Eigen DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# 瀵煎嚭锟斤拷?find_package 浣跨敤锟斤拷?Target 閰嶇疆
install(EXPORT metaSDT_Targets 
        FILE metaSDT_coreTargets.cmake 
        NAMESPACE metaSDT::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metaSDT_core)

# 鑷姩鐢熸垚 Config.cmake (璁╁閮ㄩ」鐩兘 find_package(metaSDT_core))
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/metaSDT_coreConfig.cmake"
     "include(CMakeFindDependencyMacro)\n"
     "find_dependency(OpenMP)\n"
     "find_dependency(NLopt)\n"
     "include(\"\${CMAKE_CURRENT_LIST_DIR}/metaSDT_coreTargets.cmake\")\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/metaSDT_coreConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metaSDT_core)

# 5. 鏍规嵁鏋勫缓鐜鎸傝浇鍓嶇鎷撳睍 (淇濈暀鍚庨棬鍏煎鍏ㄩ噺鏋勫缓)
if(SKBUILD) # SKBUILD ?scikit-build-core 鍚姩鏃惰嚜鍔ㄦ敞鍏ョ殑鏍囧織
    add_subdirectory(Python)
endif()
