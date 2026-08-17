"""Public Python frontend for metaSDT."""

from ._criterion import (
    criterion_likelihood,
    criterion_posterior,
    criterion_prior,
)
from ._estimate import (
    estimate_abc,
    estimate_map,
    estimate_mcmc,
    estimate_mle,
)
from ._matrix import matrix_freq, matrix_mult, matrix_prob
from ._modify import (
    info_data,
    model_bch,
    model_decay,
    model_lognormal,
    model_normal,
    model_sdt,
    modify_params,
    modify_priors,
)
from ._options import default_shell_run_m_option
from ._plot import plot_shell_run_m
from ._shell import shell_run_m
from ._ui import ui

__all__ = [
    "criterion_likelihood",
    "criterion_posterior",
    "criterion_prior",
    "default_shell_run_m_option",
    "estimate_abc",
    "estimate_map",
    "estimate_mcmc",
    "estimate_mle",
    "info_data",
    "matrix_freq",
    "matrix_mult",
    "matrix_prob",
    "model_bch",
    "model_decay",
    "model_lognormal",
    "model_normal",
    "model_sdt",
    "modify_params",
    "modify_priors",
    "plot_shell_run_m",
    "shell_run_m",
    "ui",
]
