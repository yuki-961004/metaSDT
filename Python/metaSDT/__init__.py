# 导入 C++ 编译的底层模块
import pandas
from . import _core_matrix_freq
from . import _core_matrix_prob
from . import _core_matrix_mult
from . import _core_criterion_likelihood
from . import _core_criterion_prior
from . import _core_criterion_posterior
from . import _core_data_info
from . import _help_modify_params
from . import _help_modify_prior
from . import _model_sdt
from . import _estimate_mle


def _extract_3d_mat(mat):
    """Helper to safely extract a 3D matrix from various Python inputs."""
    if isinstance(mat, pandas.DataFrame):
        return [mat.values.tolist()]
    elif isinstance(mat, dict):
        if "freq_mat" in mat:
            return mat["freq_mat"]
        if "prob_mat" in mat:
            return mat["prob_mat"]
        # Assumes a dict of DataFrames
        return [df.values.tolist() for df in mat.values()]
    return mat


def matrix_freq(stim, resp, conf=None, diff=None, std_params=None):
    """
    Calculate the frequency matrix for Signal Detection Theory.
    """
    # 1. 调用 C++ 底层核心函数，获取原始字典结果
    res = _core_matrix_freq.matrix_freq(
        stim=stim, resp=resp, conf=conf, diff=diff, std_params=std_params
    )

    if len(res["freq_mat"]) == 1:
        return pandas.DataFrame(
            res["freq_mat"][0], index=res["row_names"], columns=res["col_names"]
        )
    else:
        return {
            name: pandas.DataFrame(
                mat, index=res["row_names"], columns=res["col_names"]
            )
            for name, mat in zip(res["dim_names"], res["freq_mat"])
        }


def matrix_prob(cdf_noise, cdf_signal, std_params):
    """
    Calculate the theoretical probability matrix.
    """
    res = _core_matrix_prob.matrix_prob(
        cdf_noise=cdf_noise, cdf_signal=cdf_signal, std_params=std_params
    )
    if len(res["prob_mat"]) == 1:
        return pandas.DataFrame(
            res["prob_mat"][0], index=res["row_names"], columns=res["col_names"]
        )
    else:
        return {
            name: pandas.DataFrame(
                mat, index=res["row_names"], columns=res["col_names"]
            )
            for name, mat in zip(res["dim_names"], res["prob_mat"])
        }


def matrix_mult(freq_mat, prob_mat, std_params):
    """
    Calculate the Log-Likelihood product matrix.
    Supports both raw nested lists and pandas DataFrames.
    """
    f_mat = _extract_3d_mat(freq_mat)
    p_mat = _extract_3d_mat(prob_mat)

    # 调用底层 C++ 核心函数计算
    res_mat = _core_matrix_mult.matrix_mult(
        freq_mat=f_mat, prob_mat=p_mat, std_params=std_params
    )

    if isinstance(freq_mat, pandas.DataFrame):
        return pandas.DataFrame(
            res_mat[0], index=freq_mat.index, columns=freq_mat.columns
        )
    elif isinstance(freq_mat, dict) and not (
        "freq_mat" in freq_mat or "prob_mat" in freq_mat
    ):
        return {
            k: pandas.DataFrame(m, index=v.index, columns=v.columns)
            for (k, v), m in zip(freq_mat.items(), res_mat)
        }
    return res_mat


def criterion_likelihood(freq_mat, prob_mat, std_params):
    """
    Calculate Model Loss indicators including Negative Log-Likelihood, AIC, and BIC.
    """
    f_mat = _extract_3d_mat(freq_mat)
    p_mat = _extract_3d_mat(prob_mat)

    return _core_criterion_likelihood.criterion_likelihood(
        mult_mat=f_mat, freq_mat=p_mat, std_params=std_params
    )


def criterion_prior(user_priors, std_params=None):
    """Evaluate Log-Prior"""
    return _core_criterion_prior.criterion_prior(
        user_priors=user_priors, std_params=std_params
    )


def criterion_posterior(freq_mat, user_priors, std_params=None):
    """Evaluate Unnormalized Log-Posterior"""
    f_mat = _extract_3d_mat(freq_mat)
    return _core_criterion_posterior.criterion_posterior(
        freq_mat=f_mat, user_priors=user_priors, std_params=std_params
    )


def modify_params(user_params=None):
    """
    Modify and flatten model parameters.
    Takes a dictionary of parameters, merges them with defaults, resolves conflicts, and flattens them.
    """
    # 纯粹的 Python 壳：直接调用 C++ 底层引擎
    return _help_modify_params.modify_params(user_params)


def modify_prior(user_priors, std_params=None):
    """
    Modify and align prior distributions.
    Maps user-friendly distribution strings to the absolute indices of free parameters.
    """
    return _help_modify_prior.modify_prior(user_priors, std_params)


def model_sdt(std_params):
    """
    Evaluate SDT Model CDFs.
    Returns a dictionary containing 'cdf_noise' and 'cdf_signal'.
    """
    return _model_sdt.model_sdt(std_params)


def data_info(df, colnames=None):
    """
    Intelligently scan the dataset and extract subject-level information.
    Returns indices that can be used with pandas df.iloc[indices]
    """
    if colnames is None:
        colnames = {}

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

    return _core_data_info.data_info(df=df_dict, colnames=colnames)


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
        df=df_dict, 
        colnames=colnames, 
        params=params, 
        model=model, 
        control=control, 
        lower=lower, 
        upper=upper
    )
    if isinstance(res_list, dict):
        return {k: pandas.DataFrame(v) for k, v in res_list.items()}
    else:
        return pandas.DataFrame(res_list)


# __all__ 控制当用户使用 from metaSDT import * 时，暴露出哪些接口
__all__ = [
    # Core: 核心矩阵计算与似然操作
    "matrix_freq",
    "matrix_prob",
    "matrix_mult",
    "criterion_likelihood",
    "criterion_prior",
    "criterion_posterior",
    # Help: 参数管理与辅助工具
    "modify_params",
    "modify_prior",
    "data_info",
    # Model: 各类拓展模型引擎
    "model_sdt",
    # Fitting: 极大似然估计
    "estimate_mle",
]
