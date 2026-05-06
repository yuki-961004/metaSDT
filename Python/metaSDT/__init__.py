# 导入 C++ 编译的底层模块
import pandas
from . import _core_matrix_freq
from . import _help_modify_params
from . import _model_sdt
from . import _core_matrix_prob

def matrix_freq(stim, resp, conf=None):
    """
    Calculate the frequency matrix for Signal Detection Theory.
    """
    # 1. 调用 C++ 底层核心函数，获取原始字典结果
    res = _core_matrix_freq.matrix_freq(stim, resp, conf)
    
    # 2. 将字典无缝转换为带有原生行列名的 pandas DataFrame
    return pandas.DataFrame(
        res["freq_mat"], 
        index=res["row_names"], 
        columns=res["col_names"]
    )

def matrix_prob(cdf_noise, cdf_signal, params):
    """
    Calculate the theoretical probability matrix.
    """
    res = _core_matrix_prob.matrix_prob(cdf_noise, cdf_signal, params)
    return pandas.DataFrame(
        res["prob_mat"], 
        index=res["row_names"], 
        columns=res["col_names"]
    )

def modify_params(params=None):
    """
    Modify and flatten model parameters.
    Takes a dictionary of parameters, merges them with defaults, resolves conflicts, and flattens them.
    """
    # 纯粹的 Python 壳：直接调用 C++ 底层引擎
    return _help_modify_params.modify_params(params)

def model_sdt(params, x):
    """
    Evaluate SDT Model CDFs.
    Returns a dictionary containing 'cdf_noise' and 'cdf_signal'.
    """
    return _model_sdt.model_sdt(params, x)

# __all__ 控制当用户使用 from metaSDT import * 时，暴露出哪些接口
__all__ = ["matrix_freq", "matrix_prob", "modify_params", "model_sdt"]