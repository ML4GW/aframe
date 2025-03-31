import numpy as np
from bokeh.palettes import Bright7 as palette  # noqa
from bokeh.plotting import figure

SECONDS_PER_YEAR = 60 * 60 * 24 * 365.25


subscripts = {}
for i in range(10):
    subscripts[i] = rf"\u{2080 + i}".encode().decode("unicode-escape")


def get_figure(**kwargs):
    default_kwargs = {"height": 300, "width": 700, "tools": ""}
    kwargs = default_kwargs | kwargs
    p = figure(**kwargs)

    if not kwargs.get("tools"):
        p.toolbar_location = None

    title = kwargs.get("title")
    if title and title.startswith("$$"):
        p.title.text_font_style = "normal"
    return p


def hide_axis(p, axis) -> None:
    axis = getattr(p, axis + "axis")
    axis.major_tick_line_color = None
    axis.minor_tick_line_color = None

    # can't set this to 0 otherwise log-axis
    # plots freak out and won't render
    axis.major_label_text_font_size = "1pt"
    axis.major_label_text_color = None


def plot_err_bands(p, x, y, err, **kwargs):
    return p.patch(
        np.concatenate([x, x[::-1]]),
        np.concatenate([(y - err), (y + err)[::-1]]),
        **kwargs,
    )


def make_grid(combos):
    num_plots = len(combos)
    if num_plots not in [1, 4]:
        raise ValueError(
            f"Only support 2x2 or 1x1 grids, can't plot {num_plots} combos"
        )

    plots = []
    if num_plots == 1:
        kwargs = {
            "title": r"$$\text{{Log Normal }}m_1={}, m_2={}$$".format(
                *combos[0]
            ),
            "x_axis_type": "log",
            "tools": "save",
            "width": 380,
            "height": 260,
        }

        kwargs["x_axis_label"] = (
            r"$$\text{False Alarm Rate " r"[years}^{-1}\text{]}$$"
        )
        kwargs["x_axis_label"] = (
            r"$$\text{False Alarm Rate " r"[years}^{-1}\text{]}$$"
        )
        p = get_figure(**kwargs)
        p.outline_line_color = "#ffffff"

        plots.append(p)

        return plots

    for i, combo in enumerate(combos):
        kwargs = {
            "title": r"$$\text{{Log Normal }}m_1={}, m_2={}$$".format(*combo),
            "x_axis_type": "log",
            "tools": "save",
        }

        kwargs["width"] = 350
        if not i % 2:
            # plots on the left need space for y-axis label
            kwargs["width"] += 30
            kwargs["y_axis_label"] = (
                r"$$\text{Sensitive Volume [Gpc}" r"^{3}\text{]}$$"
            )

        kwargs["height"] = 220
        if i > 1:
            # lower plots need space for x-axis label
            kwargs["height"] += 30
            kwargs["x_axis_label"] = (
                r"$$\text{False Alarm Rate " r"[years}^{-1}\text{]}$$"
            )

        # share x range between all plots
        if plots:
            kwargs["x_range"] = plots[0].x_range

        p = get_figure(**kwargs)
        p.outline_line_color = "#ffffff"

        # don't show x axis on upper plots
        if i < 2:
            hide_axis(p, "x")
        plots.append(p)
    return plots
