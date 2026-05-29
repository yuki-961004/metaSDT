"""Model helper wrappers for the Python frontend."""

from . import _help_info_data, _help_modify_params, _help_modify_priors
from . import _model_sdt
from ._convert import to_backend_df_dict


def modify_params(user_params=None):
    return _help_modify_params.modify_params(user_params)


def modify_priors(user_priors, std_params=None):
    return _help_modify_priors.modify_priors(user_priors, std_params)


def model_sdt(std_params):
    return _model_sdt.model_sdt(std_params)


def info_data(df, colnames=None):
    if colnames is None:
        colnames = {}
    return _help_info_data.info_data(
        df=to_backend_df_dict(df),
        colnames=colnames,
    )
