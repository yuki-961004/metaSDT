#' Simulate trial-level SDT data without fitting
#'
#' @param params A structured parameter list with free, fixed, and constant.
#' @param model Model name. Version 1 supports only "sdt".
#' @param option Simulation options, including n, seed, plot, density_points,
#'   and xlim.
#' @return An S3 object of class metaSDT.shell_run_m.
#' @export
shell_run_m <- function(
    params,
    model = "sdt",
    option = list(
        n = 1000,
        seed = NULL,
        plot = TRUE,
        density_points = 512,
        xlim = NULL
    )
) {
    default_option <- list(
        n = 1000,
        seed = NULL,
        plot = TRUE,
        density_points = 512,
        xlim = NULL
    )
    option <- utils::modifyList(default_option, option)

    out <- .Call(`_metaSDT_r_shell_run_m`, params, model, option)
    out$std_params <- out$params
    out$params <- params
    class(out) <- c("metaSDT.shell_run_m", class(out))

    if (isTRUE(option$plot)) {
        plot(out)
    }

    out
}

#' @export
plot.metaSDT.shell_run_m <- function(
    x,
    ...
) {
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("Package 'ggplot2' is required for this plotting function.")
    }

    den <- x$density
    cri <- x$criteria

    p <- ggplot2::ggplot(data = den, ggplot2::aes(x = x)) +
        ggplot2::geom_area(ggplot2::aes(y = noise, fill = "Noise"), alpha = 0.3) +
        ggplot2::geom_line(ggplot2::aes(y = noise, color = "Noise"), linewidth = 1) +
        ggplot2::geom_area(ggplot2::aes(y = signal, fill = "Signal"), alpha = 0.3) +
        ggplot2::geom_line(ggplot2::aes(y = signal, color = "Signal"), linewidth = 1) +
        ggplot2::geom_vline(
            xintercept = cri,
            linetype = "dashed",
            color = "black",
            linewidth = 0.8,
            alpha = 0.7
        ) +
        ggplot2::scale_fill_manual(
            values = c("Noise" = "#3f87cf", "Signal" = "#c80154")
        ) +
        ggplot2::scale_color_manual(
            values = c("Noise" = "#3f87cf", "Signal" = "#c80154")
        ) +
        ggplot2::labs(
            title = "Signal Detection Theory Simulation",
            subtitle = paste("Model:", x$model),
            x = "Internal Response (Evidence)",
            y = "Probability Density",
            fill = "Distribution",
            color = "Distribution"
        ) +
        ggplot2::theme_minimal() +
        ggplot2::theme(
            legend.position = "top",
            plot.title = ggplot2::element_text(hjust = 0.5, face = "bold", size = 14),
            plot.subtitle = ggplot2::element_text(hjust = 0.5, size = 10, color = "gray40"),
            panel.grid.minor = ggplot2::element_blank()
        )

    print(p)
    invisible(p)
}
