#' Estimate SDT parameters with maximum likelihood
#'
#' @export
estimate_mle <- function(
    df,
    colnames = NULL,
    params = NULL,
    model = "sdt",
    control = NULL,
    lower = NULL,
    upper = NULL
) {
    .estimate_mle(df, colnames, params, model, control, lower, upper)
}

#' Estimate SDT parameters with maximum a posteriori estimation
#'
#' @export
estimate_map <- function(
    df,
    colnames = NULL,
    params = NULL,
    model = "sdt",
    control = NULL,
    lower = NULL,
    upper = NULL,
    priors = NULL
) {
    .estimate_map(
        df,
        colnames,
        params,
        model,
        control,
        lower,
        upper,
        priors
    )
}

#' Estimate SDT parameters with MCMC
#'
#' @export
estimate_mcmc <- function(
    df,
    colnames = NULL,
    params = NULL,
    model = "sdt",
    control = NULL,
    lower = NULL,
    upper = NULL,
    priors = NULL
) {
    .estimate_mcmc(
        df,
        colnames,
        params,
        model,
        control,
        lower,
        upper,
        priors
    )
}

#' Estimate SDT parameters with approximate Bayesian computation
#'
#' @export
estimate_abc <- function(
    df,
    colnames = NULL,
    params = NULL,
    model = "sdt",
    control = NULL,
    priors = NULL
) {
    .estimate_abc(df, colnames, params, model, control, priors)
}
