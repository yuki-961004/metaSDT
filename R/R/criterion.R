#' Calculate model likelihood indicators
#'
#' @param freq_mat Frequency matrix.
#' @param prob_mat Probability matrix.
#' @param std_params Standardized model parameters.
#' @return A list containing likelihood criteria such as NLL, AIC, and BIC.
#' @export
criterion_likelihood <- function(freq_mat, prob_mat, std_params) {
    .core_criterion_likelihood(freq_mat, prob_mat, std_params)
}

#' Evaluate the log prior
#'
#' @param user_priors User prior specification.
#' @param std_params Optional standardized model parameters.
#' @return Numeric log-prior value.
#' @export
criterion_prior <- function(user_priors, std_params = NULL) {
    .core_criterion_prior(user_priors, std_params)
}

#' Evaluate the log posterior
#'
#' @param freq_mat Frequency matrix.
#' @param user_priors User prior specification.
#' @param std_params Optional standardized model parameters.
#' @return Numeric log-posterior value.
#' @export
criterion_posterior <- function(freq_mat, user_priors, std_params = NULL) {
    .core_criterion_posterior(freq_mat, user_priors, std_params)
}
