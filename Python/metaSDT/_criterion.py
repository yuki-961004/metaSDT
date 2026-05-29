"""Criterion wrappers for the Python frontend."""

from . import (
    _core_criterion_likelihood,
    _core_criterion_posterior,
    _core_criterion_prior,
)
from ._convert import extract_3d_mat


def criterion_likelihood(freq_mat, prob_mat, std_params):
    """Calculate likelihood criteria for frequency and probability matrices."""
    return _core_criterion_likelihood.criterion_likelihood(
        freq_mat=extract_3d_mat(freq_mat),
        prob_mat=extract_3d_mat(prob_mat),
        std_params=std_params,
    )


def criterion_prior(user_priors, std_params=None):
    """Evaluate the log prior for a prior specification."""
    return _core_criterion_prior.criterion_prior(
        user_priors=user_priors,
        std_params=std_params,
    )


def criterion_posterior(freq_mat, user_priors, std_params=None):
    """Evaluate the log posterior for a frequency matrix and priors."""
    return _core_criterion_posterior.criterion_posterior(
        freq_mat=extract_3d_mat(freq_mat),
        user_priors=user_priors,
        std_params=std_params,
    )
