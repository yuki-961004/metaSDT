# 导入 C++ 编译的底层模块
import pandas
from . import _core

def matrix_freq(stim, resp, conf=None):
    """
    Calculate the frequency matrix for Signal Detection Theory.
    """
    # 1. 调用 C++ 底层核心函数，获取原始字典结果
    res = _core.matrix_freq(stim, resp, conf)
    
    # 2. 将字典无缝转换为带有原生行列名的 pandas DataFrame
    return pandas.DataFrame(
        res["freq_mat"], 
        index=res["row_names"], 
        columns=res["col_names"]
    )

__all__ = ["matrix_freq"]