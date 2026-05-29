"""Matrix wrappers for the Python frontend."""

import pandas

from . import _core_matrix_freq, _core_matrix_mult, _core_matrix_prob
from ._convert import extract_3d_mat


def matrix_freq(stim, resp, conf=None, diff=None, std_params=None):
    res = _core_matrix_freq.matrix_freq(
        stim=stim,
        resp=resp,
        conf=conf,
        diff=diff,
        std_params=std_params,
    )
    if len(res["freq_mat"]) == 1:
        return pandas.DataFrame(
            res["freq_mat"][0],
            index=res["row_names"],
            columns=res["col_names"],
        )
    return {
        name: pandas.DataFrame(
            mat,
            index=res["row_names"],
            columns=res["col_names"],
        )
        for name, mat in zip(res["dim_names"], res["freq_mat"])
    }


def matrix_prob(cdf_noise, cdf_signal, std_params):
    res = _core_matrix_prob.matrix_prob(
        cdf_noise=cdf_noise,
        cdf_signal=cdf_signal,
        std_params=std_params,
    )
    if len(res["prob_mat"]) == 1:
        return pandas.DataFrame(
            res["prob_mat"][0],
            index=res["row_names"],
            columns=res["col_names"],
        )
    return {
        name: pandas.DataFrame(
            mat,
            index=res["row_names"],
            columns=res["col_names"],
        )
        for name, mat in zip(res["dim_names"], res["prob_mat"])
    }


def matrix_mult(freq_mat, prob_mat, std_params):
    freq_3d = extract_3d_mat(freq_mat)
    prob_3d = extract_3d_mat(prob_mat)
    res_mat = _core_matrix_mult.matrix_mult(
        freq_mat=freq_3d,
        prob_mat=prob_3d,
        std_params=std_params,
    )
    if isinstance(freq_mat, pandas.DataFrame):
        return pandas.DataFrame(
            res_mat[0],
            index=freq_mat.index,
            columns=freq_mat.columns,
        )
    if (
        isinstance(freq_mat, dict)
        and "freq_mat" not in freq_mat
        and "prob_mat" not in freq_mat
    ):
        return {
            key: pandas.DataFrame(mat, index=value.index, columns=value.columns)
            for (key, value), mat in zip(freq_mat.items(), res_mat)
        }
    return res_mat
