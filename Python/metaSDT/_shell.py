"""Simulation shell wrappers for the Python frontend."""

import pandas

from . import _shell_run_m
from ._options import merge_shell_run_m_option
from ._plot import plot_shell_run_m


def shell_run_m(params, model="sdt", option=None):
    """Simulate trial-level SDT data without estimating parameters.

    Parameters
    ----------
    params : dict
        Model parameter specification. A structured dict may contain
        ``"free"``, ``"fixed"``, and ``"constant"`` entries. A flat dict is
        interpreted by the backend as free parameters.
    model : str, default "sdt"
        Model name. Currently, ``"sdt"`` is the main supported path.
    option : dict or None
        Shell-level options. Supported keys are ``"n"``, ``"seed"``,
        ``"plot"``, ``"density_points"``, ``"xlim"``, and ``"show"``.

    Returns
    -------
    dict
        A result dict containing ``params``, ``model``, ``data``, ``density``,
        ``criteria``, ``option``, ``seed``, and ``plot`` when requested.

    Notes
    -----
    ``shell_run_m`` simulates data and does not fit parameters. Its ``option``
    argument is separate from estimator ``control`` settings.
    """
    merged_option = merge_shell_run_m_option(option)

    result = _shell_run_m.shell_run_m(
        params=params,
        model=model,
        option=merged_option,
    )
    result["std_params"] = result["params"]
    result["params"] = params
    result["data"] = pandas.DataFrame(result["data"])
    result["density"] = pandas.DataFrame(result["density"])
    result["option"] = dict(merged_option)

    if merged_option["plot"]:
        result["plot"] = plot_shell_run_m(
            result,
            model_name=model,
            show=merged_option["show"],
        )

    return result
