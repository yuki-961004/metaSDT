#' Simulate trial-level SDT data without fitting
#'
#' `shell_run_m()` is a high-level simulation shell for metaSDT workflows. It
#' simulates trial-level data from model parameters and does not estimate
#' parameters. It is intended as Step 1 of the future metaSDT modeling workflow:
#' generate and inspect model-implied data before fitting estimators.
#'
#' The `option` argument controls shell-level simulation and display behavior,
#' such as the number of simulated trials and whether to draw the density plot.
#' It is separate from estimator `control`, which is reserved for algorithm
#' settings in functions such as [estimate_mle()] and [estimate_map()].
#'
#' Currently, `"sdt"` is the main supported model path.
#'
#' @param params A structured parameter list with `free`, `fixed`, and
#'   `constant` entries, or a flat named list interpreted as free parameters.
#' @param model Model name. Currently, `"sdt"` is the main supported path.
#' @param option Shell-level options. See [default_shell_run_m_option()].
#' @return An S3 object with class `"metaSDT.shell_run_m"`.
#' @examples
#' fit <- shell_run_m(
#'   params = list(
#'     free = list(d = 1.5, c_resp = 0.0, c_conf = c(0.5, 1.0, 1.5)),
#'     fixed = list(sd_signal = 1.0, sd_noise = 1.0)
#'   ),
#'   model = "sdt",
#'   option = list(n = 1000, seed = 123, plot = FALSE)
#' )
#' @export
shell_run_m <- function(params, model = "sdt", option = list()) {
    merged_option <- merge_shell_run_m_option(option)

    out <- .shell_run_m(params, model, merged_option)
    out$std_params <- out$params
    out$params <- params
    out$option <- merged_option
    class(out) <- c("metaSDT.shell_run_m", class(out))

    if (isTRUE(merged_option$plot)) {
        plot(out, show = isTRUE(merged_option$show))
    }

    out
}

#' Print a shell_run_m simulation
#'
#' @param x A `"metaSDT.shell_run_m"` object.
#' @param ... Unused.
#' @return The input object, invisibly.
#' @export
print.metaSDT.shell_run_m <- function(x, ...) {
    n_trials <- if (!is.null(x$data)) {
        nrow(x$data)
    } else {
        NA_integer_
    }
    seed_text <- if (!is.null(x$seed)) {
        as.character(x$seed)
    } else {
        "NA"
    }

    cat("metaSDT shell_run_m simulation\n")
    cat("  model: ", x$model, "\n", sep = "")
    cat("  trials: ", n_trials, "\n", sep = "")
    cat("  seed: ", seed_text, "\n", sep = "")
    cat("  density: ", !is.null(x$density), "\n", sep = "")
    cat("  criteria: ", !is.null(x$criteria), "\n", sep = "")

    invisible(x)
}

#' Plot a shell_run_m simulation
#'
#' @param x A `"metaSDT.shell_run_m"` object.
#' @param show Whether to print the plot.
#' @param ... Unused.
#' @return A ggplot object, invisibly.
#' @importFrom ggplot2 aes element_blank element_text geom_area geom_line
#'   geom_vline ggplot labs scale_color_manual scale_fill_manual theme
#'   theme_minimal
#' @export
plot.metaSDT.shell_run_m <- function(x, show = TRUE, ...) {
    if (is.null(x$density) || is.null(x$criteria)) {
        stop(
            "shell_run_m result must contain density and criteria to plot.",
            call. = FALSE
        )
    }

    density <- x$density
    criteria <- x$criteria

    plot_obj <- ggplot2::ggplot(data = density, ggplot2::aes(x = x)) +
        ggplot2::geom_area(
            ggplot2::aes(y = noise, fill = "Noise"),
            alpha = 0.3
        ) +
        ggplot2::geom_line(
            ggplot2::aes(y = noise, color = "Noise"),
            linewidth = 1
        ) +
        ggplot2::geom_area(
            ggplot2::aes(y = signal, fill = "Signal"),
            alpha = 0.3
        ) +
        ggplot2::geom_line(
            ggplot2::aes(y = signal, color = "Signal"),
            linewidth = 1
        ) +
        ggplot2::geom_vline(
            xintercept = criteria,
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
            plot.title = ggplot2::element_text(
                hjust = 0.5,
                face = "bold",
                size = 14
            ),
            plot.subtitle = ggplot2::element_text(
                hjust = 0.5,
                size = 10,
                color = "gray40"
            ),
            panel.grid.minor = ggplot2::element_blank()
        )

    if (isTRUE(show)) {
        print(plot_obj)
    }

    invisible(plot_obj)
}
