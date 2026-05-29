"""Estimator wrappers for the Python frontend."""

from . import _estimate_abc, _estimate_map, _estimate_mcmc, _estimate_mle
from ._convert import as_meta_result, to_backend_df_dict


def estimate_mle(
    df,
    colnames=None,
    params=None,
    model="sdt",
    control=None,
    lower=None,
    upper=None,
):
    """Estimate SDT parameters using maximum likelihood."""
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
    result = _estimate_mle.estimate_mle(
        df=to_backend_df_dict(df),
        colnames=colnames,
        params=params,
        model=model,
        control=control,
        lower=lower,
        upper=upper,
    )
    return as_meta_result(result, estimator_name="MLE", control=control)


def estimate_map(
    df,
    colnames=None,
    params=None,
    model="sdt",
    control=None,
    lower=None,
    upper=None,
    priors=None,
):
    """Estimate SDT parameters using maximum a posteriori estimation."""
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
    if priors is None:
        priors = {}

    result = _estimate_map.estimate_map(
        df=to_backend_df_dict(df),
        colnames=colnames,
        params=params,
        model=model,
        control=control,
        lower=lower,
        upper=upper,
        priors=priors,
    )
    return as_meta_result(result, estimator_name="MAP", control=control)


def estimate_mcmc(
    df,
    colnames=None,
    params=None,
    model="sdt",
    control=None,
    lower=None,
    upper=None,
    priors=None,
):
    """Estimate SDT parameters using MCMC."""
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
    if priors is None:
        priors = {}

    result = _estimate_mcmc.estimate_mcmc(
        df=to_backend_df_dict(df),
        colnames=colnames,
        params=params,
        model=model,
        control=control,
        lower=lower,
        upper=upper,
        priors=priors,
    )
    return as_meta_result(result, estimator_name="MCMC", control=control)


def estimate_abc(
    df,
    colnames=None,
    params=None,
    model="sdt",
    control=None,
    priors=None,
):
    """Estimate SDT parameters using approximate Bayesian computation."""
    if colnames is None:
        colnames = {}
    if params is None:
        params = {}
    if control is None:
        control = {}
    if priors is None:
        priors = {}

    result = _estimate_abc.estimate_abc(
        df=to_backend_df_dict(df),
        colnames=colnames,
        params=params,
        model=model,
        control=control,
        priors=priors,
    )
    return as_meta_result(result, estimator_name="ABC", control=control)
