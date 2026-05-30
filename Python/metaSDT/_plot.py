"""Plotting utilities for shell_run_m results."""

import pandas


def plot_shell_run_m(result, model_name=None, show=True):
    """Create a plotnine density plot for a ``shell_run_m`` result.

    Parameters
    ----------
    result : dict
        A result returned by :func:`metaSDT.shell_run_m`.
    model_name : str or None
        Optional model name for the plot subtitle. When omitted, the value in
        ``result["model"]`` is used.
    show : bool, default True
        Whether to print the plot object for display in interactive sessions.

    Returns
    -------
    plotnine.ggplot.ggplot
        The plotnine plot object.
    """
    try:
        from plotnine import aes
        from plotnine import element_blank
        from plotnine import element_text
        from plotnine import geom_area
        from plotnine import geom_line
        from plotnine import geom_vline
        from plotnine import ggplot
        from plotnine import labs
        from plotnine import scale_color_manual
        from plotnine import scale_fill_manual
        from plotnine import theme
        from plotnine import theme_minimal
    except ImportError as exc:
        raise ImportError(
            "plotnine is required for plot_shell_run_m(). "
            "Install it with: pip install plotnine."
        ) from exc

    density = result["density"]
    criteria = result["criteria"]
    subtitle_model = model_name if model_name is not None else result["model"]

    density_long = pandas.melt(
        density,
        id_vars=["x"],
        value_vars=["noise", "signal"],
        var_name="Distribution",
        value_name="density_val",
    )
    density_long["Distribution"] = (
        density_long["Distribution"].str.capitalize()
    )

    colors = {"Noise": "#3f87cf", "Signal": "#c80154"}
    plot_obj = (
        ggplot(
            density_long,
            aes(
                x="x",
                y="density_val",
                fill="Distribution",
                color="Distribution",
            ),
        )
        + geom_area(alpha=0.3, position="identity")
        + geom_line(size=1)
        + geom_vline(
            xintercept=criteria,
            linetype="dashed",
            color="black",
            size=0.8,
            alpha=0.7,
        )
        + scale_fill_manual(values=colors)
        + scale_color_manual(values=colors)
        + labs(
            title="Signal Detection Theory Simulation",
            subtitle=f"Model: {subtitle_model}",
            x="Internal Response (Evidence)",
            y="Probability Density",
        )
        + theme_minimal()
        + theme(
            legend_position="top",
            plot_title=element_text(hjust=0.5, weight="bold", size=14),
            plot_subtitle=element_text(hjust=0.5, size=10, color="dimgray"),
            panel_grid_minor=element_blank(),
            figure_size=(8, 5),
        )
    )

    if show:
        plot_obj.show()

    return plot_obj
