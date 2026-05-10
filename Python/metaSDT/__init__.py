# 导入 C++ 编译的底层模块
import pandas
from . import _core_matrix_freq
from . import _core_matrix_prob
from . import _core_matrix_mult
from . import _core_loss_function
from . import _core_data_info
from . import _help_modify_params
from . import _model_sdt
from . import _estimate_mle


def matrix_freq(stim, resp, conf=None):
    """
    Calculate the frequency matrix for Signal Detection Theory.
    """
    # 1. 调用 C++ 底层核心函数，获取原始字典结果
    res = _core_matrix_freq.matrix_freq(stim, resp, conf)

    # 2. 将字典无缝转换为带有原生行列名的 pandas DataFrame
    return pandas.DataFrame(
        res["freq_mat"], index=res["row_names"], columns=res["col_names"]
    )


def matrix_prob(cdf_noise, cdf_signal, params):
    """
    Calculate the theoretical probability matrix.
    """
    res = _core_matrix_prob.matrix_prob(cdf_noise, cdf_signal, params)
    return pandas.DataFrame(
        res["prob_mat"], index=res["row_names"], columns=res["col_names"]
    )


def matrix_mult(freq_mat, prob_mat, params):
    """
    Calculate the Log-Likelihood product matrix.
    Supports both raw nested lists and pandas DataFrames.
    """
    # 如果输入是 DataFrame，提取为纯嵌套列表供 C++ 使用，并保留原始维度名称
    if isinstance(freq_mat, pandas.DataFrame):
        f_mat = freq_mat.values.tolist()
        idx = freq_mat.index
        cols = freq_mat.columns
    else:
        f_mat = freq_mat
        idx, cols = None, None

    p_mat = (
        prob_mat.values.tolist() if isinstance(prob_mat, pandas.DataFrame) else prob_mat
    )

    # 调用底层 C++ 核心函数计算
    res_mat = _core_matrix_mult.matrix_mult(f_mat, p_mat, params)

    # 如果原本输入的是 DataFrame，则保持原样返回继承了行列名的 DataFrame
    if idx is not None and cols is not None:
        return pandas.DataFrame(res_mat, index=idx, columns=cols)
    return res_mat


def loss_function(freq_mat, prob_mat, params):
    """
    Calculate Model Loss indicators including Negative Log-Likelihood, AIC, and BIC.
    """
    # 安全地将 DataFrame 转为 C++ 能够接收的嵌套列表
    f_mat = (
        freq_mat.values.tolist() if isinstance(freq_mat, pandas.DataFrame) else freq_mat
    )
    p_mat = (
        prob_mat.values.tolist() if isinstance(prob_mat, pandas.DataFrame) else prob_mat
    )

    return _core_loss_function.loss_function(f_mat, p_mat, params)


def modify_params(params=None):
    """
    Modify and flatten model parameters.
    Takes a dictionary of parameters, merges them with defaults, resolves conflicts, and flattens them.
    """
    # 纯粹的 Python 壳：直接调用 C++ 底层引擎
    return _help_modify_params.modify_params(params)


def model_sdt(params):
    """
    Evaluate SDT Model CDFs.
    Returns a dictionary containing 'cdf_noise' and 'cdf_signal'.
    """
    return _model_sdt.model_sdt(params)


def data_info(df, colnames=None, condition=None):
    """
    Intelligently scan the dataset and extract subject-level information.
    Returns indices that can be used with pandas df.iloc[indices]
    """
    if colnames is None:
        colnames = {}
    if condition is None:
        condition = []
    elif isinstance(condition, str):
        condition = [condition]

    # 将 pandas DataFrame 的每一列安全转变为可以传入 C++ 的浮点数组
    df_dict = {}
    for col in df.columns:
        if pandas.api.types.is_numeric_dtype(df[col]) or pandas.api.types.is_bool_dtype(
            df[col]
        ):
            df_dict[str(col)] = df[col].astype(float).tolist()
        else:
            # 将纯字符串列 (如 'Easy') 转为因子数字编码 (0, 1, 2...) 以防它们是分组条件
            codes, _ = pandas.factorize(df[col])
            df_dict[str(col)] = codes.astype(float).tolist()

    return _core_data_info.data_info(df_dict, colnames, condition)


def estimate_mle(
    df,
    colnames=None,
    params=None,
    model="sdt",
    control=None,
    lower=None,
    upper=None,
):
    """
    Perform Maximum Likelihood Estimation (MLE) optimization.
    Automatically converts pandas DataFrame for the C++ backend.
    """
    if colnames is None:
        colnames = {}
    if params is None:
        params = {}
    if control is None:
        control = {}
    if lower is None:
        lower = {}
    if upper is None:
        upper = {}

    # 将 pandas DataFrame 的每一列安全转变为可以传入 C++ 的浮点数组
    df_dict = {}
    for col in df.columns:
        if pandas.api.types.is_numeric_dtype(df[col]) or pandas.api.types.is_bool_dtype(
            df[col]
        ):
            df_dict[str(col)] = df[col].astype(float).tolist()
        else:
            codes, _ = pandas.factorize(df[col])
            df_dict[str(col)] = codes.astype(float).tolist()

    # 调用底层 C++ 并释放 GIL 进行多线程狂飙
    res_list = _estimate_mle.estimate_mle(
        df_dict, colnames, params, model, control, lower, upper
    )
    return pandas.DataFrame(res_list)


# __all__ 控制当用户使用 from metaSDT import * 时，暴露出哪些接口
__all__ = [
    # Core: 核心矩阵计算与似然操作
    "matrix_freq",
    "matrix_prob",
    "matrix_mult",
    "loss_function",
    # Help: 参数管理与辅助工具
    "modify_params",
    "data_info",
    # Model: 各类拓展模型引擎
    "model_sdt",
    # Fitting: 极大似然估计
    "estimate_mle",
]
