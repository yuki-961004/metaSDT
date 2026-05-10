from setuptools import setup, Extension
import sys


class get_pybind_include(object):
    """延迟导入 pybind11 获取路径，防止在 setup 环境初始化前报错"""

    def __str__(self):
        import pybind11

        return pybind11.get_include()


# 针对不同系统 (Windows vs Linux/Mac) 指定 C++17 编译参数
extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]

ext_modules = [
    # ==============================================================================
    # 1. Core Modules: 核心矩阵计算与似然操作
    # ==============================================================================
    Extension(
        "metaSDT._core_matrix_freq",
        ["src/wrapper_matrix_freq.cpp"],
        include_dirs=[get_pybind_include(), "src/Cpp/include"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "metaSDT._core_matrix_prob",
        ["src/wrapper_matrix_prob.cpp"],
        include_dirs=[get_pybind_include(), "src/Cpp/include"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "metaSDT._core_matrix_mult",
        ["src/wrapper_matrix_mult.cpp"],
        include_dirs=[get_pybind_include(), "src/Cpp/include"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "metaSDT._core_loss_function",
        ["src/wrapper_loss_function.cpp"],
        include_dirs=[get_pybind_include(), "src/Cpp/include"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    # ==============================================================================
    # 2. Help Modules: 参数管理与辅助工具
    # ==============================================================================
    Extension(
        "metaSDT._help_modify_params",
        ["src/wrapper_modify_params.cpp"],
        include_dirs=[get_pybind_include(), "src/Cpp/include"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "metaSDT._core_data_info",
        ["src/wrapper_data_info.cpp"],
        include_dirs=[get_pybind_include(), "src/Cpp/include"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    # ==============================================================================
    # 3. Model Modules: SDT 及各类拓展模型引擎
    # ==============================================================================
    Extension(
        "metaSDT._model_sdt",
        ["src/wrapper_model_sdt.cpp"],
        include_dirs=[get_pybind_include(), "src/Cpp/include"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "metaSDT._estimate_mle",
        ["src/wrapper_estimate_mle.cpp"],
        include_dirs=[get_pybind_include(), "src/Cpp/include"],
        libraries=["nlopt"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="metaSDT",
    version="0.0.3",
    ext_modules=ext_modules,
)
