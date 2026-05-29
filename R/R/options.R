#' Default options for shell_run_m
#'
#' @return A list of shell-level simulation and display options.
#' @export
default_shell_run_m_option <- function() {
    list(
        n = 1000L,
        seed = NULL,
        plot = TRUE,
        density_points = 512L,
        xlim = NULL,
        show = TRUE
    )
}

merge_shell_run_m_option <- function(option = list()) {
    if (is.null(option)) {
        option <- list()
    }
    if (!is.list(option)) {
        stop("option must be a list or NULL.", call. = FALSE)
    }

    default_option <- default_shell_run_m_option()
    unknown_names <- setdiff(names(option), names(default_option))
    if (length(unknown_names) > 0L) {
        warning(
            "Unknown shell_run_m option field(s): ",
            paste(unknown_names, collapse = ", "),
            call. = FALSE
        )
    }

    utils::modifyList(default_option, option)
}
