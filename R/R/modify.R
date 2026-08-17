#' Intelligently scan a data set and extract subject-level information
#'
#' @param df Data frame.
#' @param colnames Optional column-name mapping.
#' @return Subject-level data information.
#' @export
info_data <- function(df, colnames = NULL) {
    .help_info_data(df, colnames)
}

#' Modify and flatten model parameters
#'
#' @param user_params Optional user parameter specification.
#' @return Standardized and flattened model parameters.
#' @export
modify_params <- function(user_params = NULL) {
    .help_modify_params(user_params)
}

#' Modify and align prior distributions
#'
#' @param user_priors User prior specification.
#' @param std_params Optional standardized model parameters.
#' @return Standardized prior information.
#' @export
modify_priors <- function(user_priors, std_params = NULL) {
    .help_modify_priors(user_priors, std_params)
}

#' Evaluate the SDT model CDFs
#'
#' @param params Standardized model parameters.
#' @return Noise and signal CDF values.
#' @export
model_sdt <- function(params) {
    .core_model_sdt(params)
}

#' Evaluate the BCH model CDFs
#'
#' @param params Standardized model parameters.
#' @return Noise and signal CDF values and criteria.
#' @export
model_bch <- function(params) {
    .core_model_bch(params)
}

#' Evaluate the Normal meta-noise model probabilities
#'
#' @param params Standardized model parameters.
#' @return Model response-confidence probabilities.
#' @export
model_normal <- function(params) {
    .core_model_normal(params)
}

#' Evaluate the Log-normal meta-noise model probabilities
#'
#' @param params Standardized model parameters.
#' @return Model response-confidence probabilities.
#' @export
model_lognormal <- function(params) {
    .core_model_lognormal(params)
}

#' Evaluate the Signal Decay model probabilities
#'
#' @param params Standardized model parameters.
#' @return Model response-confidence probabilities.
#' @export
model_decay <- function(params) {
    .core_model_decay(params)
}
