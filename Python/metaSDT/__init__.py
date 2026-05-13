import pandas
from . import _core_matrix_freq, _core_matrix_prob, _core_matrix_mult
from . import _core_criterion_likelihood, _core_criterion_prior, _core_criterion_posterior
from . import _help_data_info, _help_modify_params, _help_modify_prior
from . import _model_sdt, _estimate_mle, _estimate_map, _ui


class _UI:
    @staticmethod
    def progress_start(total, title="Progress", refresh_ms=100, mode="auto", line_interval_sec=2, line_interval_pct=5.0):
        _ui.progress_start(
            total=total,
            title=title,
            refresh_ms=refresh_ms,
            mode=mode,
            line_interval_sec=line_interval_sec,
            line_interval_pct=line_interval_pct,
        )

    @staticmethod
    def progress_bar(current):
        _ui.progress_bar(current=current)

    @staticmethod
    def progress_advance(step=1):
        _ui.progress_advance(step=step)

    @staticmethod
    def progress_finish():
        _ui.progress_finish()

    @staticmethod
    def progress_snapshot():
        return _ui.progress_snapshot()


ui = _UI()


def _extract_3d_mat(mat):
    if isinstance(mat, pandas.DataFrame):
        return [mat.values.tolist()]
    if isinstance(mat, dict):
        if "freq_mat" in mat:
            return mat["freq_mat"]
        if "prob_mat" in mat:
            return mat["prob_mat"]
        return [df.values.tolist() for df in mat.values()]
    return mat


def _to_backend_df_dict(df):
    out = {}
    for col in df.columns:
        if pandas.api.types.is_numeric_dtype(df[col]) or pandas.api.types.is_bool_dtype(df[col]):
            out[str(col)] = df[col].astype(float).tolist()
        else:
            codes, _ = pandas.factorize(df[col])
            out[str(col)] = codes.astype(float).tolist()
    return out


def _coerce_fit(fit):
    if isinstance(fit, list):
        return pandas.DataFrame(fit)
    if isinstance(fit, dict):
        return {k: pandas.DataFrame(v) if isinstance(v, list) else v for k, v in fit.items()}
    return fit


def _normalize_condition_keys(x, prefix="condition"):
    if not isinstance(x, dict):
        return x
    out = {}
    for i, (k, v) in enumerate(x.items()):
        key = str(k) if k is not None else ""
        key = key.strip()
        if key == "":
            key = f"{prefix}_{i}"
        out[key] = v
    return out


def _build_estimator_block(estimator_name, control):
    control_used = dict(control) if isinstance(control, dict) else {}
    return {
        "name": estimator_name,
        "backend": "nlopt",
        "global_algorithm": control_used.get("algorithm"),
        "local_algorithm": control_used.get("local_algorithm"),
        "control": control_used,
    }


def _as_meta_result(res, estimator_name, control):
    if isinstance(res, dict) and {"fit", "estimator", "diagnostics"}.issubset(res.keys()):
        out = dict(res)
        out["fit"] = _coerce_fit(out["fit"])
        if isinstance(out["fit"], dict):
            out["fit"] = _normalize_condition_keys(out["fit"])
            if isinstance(out["diagnostics"], dict):
                out["diagnostics"] = _normalize_condition_keys(out["diagnostics"])
        return out

    fit = _coerce_fit(res)
    diagnostics = {}
    if isinstance(fit, dict):
        fit = _normalize_condition_keys(fit)
        diagnostics = {k: {} for k in fit.keys()}

    return {
        "fit": fit,
        "estimator": _build_estimator_block(estimator_name, control),
        "diagnostics": diagnostics,
    }


def matrix_freq(stim, resp, conf=None, diff=None, std_params=None):
    res = _core_matrix_freq.matrix_freq(stim=stim, resp=resp, conf=conf, diff=diff, std_params=std_params)
    if len(res["freq_mat"]) == 1:
        return pandas.DataFrame(res["freq_mat"][0], index=res["row_names"], columns=res["col_names"])
    return {name: pandas.DataFrame(mat, index=res["row_names"], columns=res["col_names"]) for name, mat in zip(res["dim_names"], res["freq_mat"])}


def matrix_prob(cdf_noise, cdf_signal, std_params):
    res = _core_matrix_prob.matrix_prob(cdf_noise=cdf_noise, cdf_signal=cdf_signal, std_params=std_params)
    if len(res["prob_mat"]) == 1:
        return pandas.DataFrame(res["prob_mat"][0], index=res["row_names"], columns=res["col_names"])
    return {name: pandas.DataFrame(mat, index=res["row_names"], columns=res["col_names"]) for name, mat in zip(res["dim_names"], res["prob_mat"])}


def matrix_mult(freq_mat, prob_mat, std_params):
    f_mat = _extract_3d_mat(freq_mat)
    p_mat = _extract_3d_mat(prob_mat)
    res_mat = _core_matrix_mult.matrix_mult(freq_mat=f_mat, prob_mat=p_mat, std_params=std_params)
    if isinstance(freq_mat, pandas.DataFrame):
        return pandas.DataFrame(res_mat[0], index=freq_mat.index, columns=freq_mat.columns)
    if isinstance(freq_mat, dict) and not ("freq_mat" in freq_mat or "prob_mat" in freq_mat):
        return {k: pandas.DataFrame(m, index=v.index, columns=v.columns) for (k, v), m in zip(freq_mat.items(), res_mat)}
    return res_mat


def criterion_likelihood(freq_mat, prob_mat, std_params):
    return _core_criterion_likelihood.criterion_likelihood(freq_mat=_extract_3d_mat(freq_mat), prob_mat=_extract_3d_mat(prob_mat), std_params=std_params)


def criterion_prior(user_priors, std_params=None):
    return _core_criterion_prior.criterion_prior(user_priors=user_priors, std_params=std_params)


def criterion_posterior(freq_mat, user_priors, std_params=None):
    return _core_criterion_posterior.criterion_posterior(freq_mat=_extract_3d_mat(freq_mat), user_priors=user_priors, std_params=std_params)


def modify_params(user_params=None):
    return _help_modify_params.modify_params(user_params)


def modify_prior(user_priors, std_params=None):
    return _help_modify_prior.modify_prior(user_priors, std_params)


def model_sdt(std_params):
    return _model_sdt.model_sdt(std_params)


def data_info(df, colnames=None):
    if colnames is None:
        colnames = {}
    return _help_data_info.data_info(df=_to_backend_df_dict(df), colnames=colnames)


def estimate_mle(df, colnames=None, params=None, model="sdt", control=None, lower=None, upper=None):
    if colnames is None: colnames = {}
    if params is None: params = {}
    if control is None: control = {}
    if lower is None: lower = {}
    if upper is None: upper = {}
    res = _estimate_mle.estimate_mle(df=_to_backend_df_dict(df), colnames=colnames, params=params, model=model, control=control, lower=lower, upper=upper)
    return _as_meta_result(res, estimator_name="MLE", control=control)


def estimate_map(df, colnames=None, params=None, model="sdt", control=None, lower=None, upper=None, user_priors=None):
    if colnames is None: colnames = {}
    if params is None: params = {}
    if control is None: control = {}
    if lower is None: lower = {}
    if upper is None: upper = {}
    if user_priors is None: user_priors = {}
    res = _estimate_map.estimate_map(df=_to_backend_df_dict(df), colnames=colnames, params=params, model=model, control=control, lower=lower, upper=upper, user_priors=user_priors)
    return _as_meta_result(res, estimator_name="MAP", control=control)


__all__ = [
    "matrix_freq", "matrix_prob", "matrix_mult",
    "criterion_likelihood", "criterion_prior", "criterion_posterior",
    "modify_params", "modify_prior", "data_info",
    "model_sdt", "estimate_mle", "estimate_map", "ui",
]
