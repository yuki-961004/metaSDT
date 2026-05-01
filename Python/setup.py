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
    Extension(
        "metaSDT._core",
        ["src/wrapper_matrix_freq.cpp"],
        include_dirs=[
            get_pybind_include(),
            "src/Cpp/include"
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="metaSDT",
    version="0.0.0",
    ext_modules=ext_modules,
)