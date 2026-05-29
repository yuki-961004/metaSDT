#' Calculate the frequency matrix for Signal Detection Theory
#'
#' @param stim Numeric vector indicating the signal type.
#' @param resp Numeric vector indicating the response.
#' @param conf Optional numeric vector indicating confidence.
#' @param diff Optional numeric vector indicating difficulty.
#' @return A list-backed frequency matrix from the C++ backend.
#' @export
matrix_freq <- function(stim, resp, conf = NULL, diff = NULL) {
    .core_matrix_freq(stim, resp, conf, diff)
}

#' Calculate the model probability matrix
#'
#' @param cdf_noise Noise cumulative distribution values.
#' @param cdf_signal Signal cumulative distribution values.
#' @param params Standardized model parameters.
#' @return A list-backed probability matrix from the C++ backend.
#' @export
matrix_prob <- function(cdf_noise, cdf_signal, params) {
    .core_matrix_prob(cdf_noise, cdf_signal, params)
}

#' Calculate the log-likelihood product matrix
#'
#' @param freq_mat Frequency matrix.
#' @param prob_mat Probability matrix.
#' @param std_params Standardized model parameters.
#' @return A matrix-like list used by likelihood calculations.
#' @export
matrix_mult <- function(freq_mat, prob_mat, std_params) {
    .core_matrix_mult(freq_mat, prob_mat, std_params)
}
