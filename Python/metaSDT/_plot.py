import warnings
import pandas as pd

def plot_shell_run_m(res, model_name="sdt"):
    """
    A beautiful, ggplot2-like visualization for Signal Detection Theory simulations.
    """
    try:
        from plotnine import (
            ggplot, aes, geom_area, geom_line, geom_vline, scale_fill_manual,
            scale_color_manual, labs, theme_minimal, theme, element_text, element_blank
        )
    except ImportError:
        warnings.warn(
            "plotnine is not installed, so shell_run_m did not plot. "
            "Try running: pip install plotnine",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    density = res["density"]
    criteria = res["criteria"]

    # 将宽格式数据转换为长格式，这是 ggplot2/plotnine 的标准画图数据格式
    df_long = pd.melt(
        density, 
        id_vars=["x"], 
        value_vars=["noise", "signal"],
        var_name="Distribution", 
        value_name="density_val"
    )
    # 将标签首字母大写，用于完美展示图例
    df_long["Distribution"] = df_long["Distribution"].str.capitalize()

    # 配置新颜色：Noise 为蓝，Signal 为红
    colors = {"Noise": "#3f87cf", "Signal": "#c80154"}

    p = (
        ggplot(df_long, aes(x="x", y="density_val", fill="Distribution", color="Distribution"))
        + geom_area(alpha=0.3, position="identity")
        + geom_line(size=1)
        + geom_vline(xintercept=criteria, linetype="dashed", color="black", size=0.8, alpha=0.7)
        + scale_fill_manual(values=colors)
        + scale_color_manual(values=colors)
        + labs(
            title="Signal Detection Theory Simulation",
            subtitle=f"Model: {model_name}",
            x="Internal Response (Evidence)",
            y="Probability Density",
        )
        + theme_minimal()
        + theme(
            legend_position="top",
            plot_title=element_text(hjust=0.5, weight="bold", size=14),
            plot_subtitle=element_text(hjust=0.5, size=10, color="dimgray"),
            panel_grid_minor=element_blank(),
            figure_size=(8, 5)
        )
    )
    
    # 显式生成 matplotlib 的 Figure 对象
    fig = p.draw()
    # 挂起并显示窗口，防止普通的 .py 脚本执行完毕后瞬间关闭导致看不见图
    import matplotlib.pyplot as plt
    plt.show()