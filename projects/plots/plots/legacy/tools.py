from typing import Optional

import numpy as np
from bokeh.palettes import Bright7 as palette  # noqa
from bokeh.plotting import figure
from scipy.integrate import quad

SECONDS_PER_MONTH = 3600 * 24 * 30


subscripts = {}
for i in range(10):
    subscripts[i] = rf"\u{2080 + i}".encode().decode("unicode-escape")


def volume_element(cosmology, z):
    return cosmology.differential_comoving_volume(z).value / (1 + z)


def get_astrophysical_volume(
    zmin: float,
    zmax: float,
    cosmology,
    dec_range: Optional[tuple[float, float]] = None,
):
    volume, _ = quad(lambda z: volume_element(cosmology, z), zmin, zmax)
    if dec_range is not None:
        decmin, decmax = dec_range
        theta_max = np.pi / 2 - decmin
        theta_min = np.pi / 2 - decmax
    else:
        theta_min, theta_max = 0, np.pi
    omega = -2 * np.pi * (np.cos(theta_max) - np.cos(theta_min))
    return volume * omega


def get_figure(**kwargs):
    default_kwargs = dict(height=300, width=700, tools="")
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
        kwargs = dict(
            title=r"$$\text{{Log Normal }}m_1={}, m_2={}$$".format(*combos[0]),
            x_axis_type="log",
            tools="save",
            width=380,
            height=260,
        )

        kwargs["x_axis_label"] = (
            r"$$\text{False Alarm Rate " r"[months}^{-1}\text{]}$$"
        )
        kwargs["x_axis_label"] = (
            r"$$\text{False Alarm Rate " r"[months}^{-1}\text{]}$$"
        )
        p = get_figure(**kwargs)
        p.outline_line_color = "#ffffff"

        plots.append(p)

        return plots

    for i, combo in enumerate(combos):
        kwargs = dict(
            title=r"$$\text{{Log Normal }}m_1={}, m_2={}$$".format(*combo),
            x_axis_type="log",
            tools="save",
        )

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
                r"$$\text{False Alarm Rate " r"[months}^{-1}\text{]}$$"
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
