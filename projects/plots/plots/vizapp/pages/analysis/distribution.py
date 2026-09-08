import logging

import numpy as np
from bokeh.layouts import Spacer, column, row
from bokeh.models import (
    BooleanFilter,
    BoxSelectTool,
    Button,
    CDSView,
    ColumnDataSource,
    HoverTool,
    LogAxis,
    PanTool,
    Range1d,
    RangeSlider,
    TapTool,
    WheelZoomTool,
)
from bokeh.plotting import figure

from plots.core.style import analysis_palette as palette

FORE_ATTRS = [
    "shift",
    "mass_1",
    "mass_2",
    "mass_1_source",
    "mass_2_source",
    "snr",
    "detection_statistic",
    "injection_time",
    "chirp_mass",
]

SLIDER_ATTRS = ["mass_1_source", "mass_2_source", "snr"]

MAX_SCATTER_POINTS = 5_000
MAX_FOREGROUND_POINTS = 20_000


class DistributionPlot:
    def __init__(self, page, event_inspector) -> None:
        self.page = page
        self.event_inspector = event_inspector
        self.bckgd_color = palette[4]
        self.frgd_color = palette[2]

    def asdict(self, foreground, idx=slice(None)):
        _foreground = {
            attr: getattr(foreground, attr)[idx] for attr in FORE_ATTRS
        }
        ifo_snrs = foreground.ifo_snrs[idx]
        for i, ifo in enumerate(foreground.ifos):
            _foreground[f"{ifo}_snr"] = ifo_snrs[:, i]
        sorted_snrs = np.sort(ifo_snrs, axis=-1)
        _foreground["snr_ratio"] = sorted_snrs[:, -1] / sorted_snrs[:, -2]
        return _foreground

    def initialize_sources(self):
        self.bar_source = ColumnDataSource(
            {"center": [], "top": [], "width": []}
        )
        self.background_source = ColumnDataSource(
            {
                "x": [],
                "detection_time": [],
                "detection_statistic": [],
                "shifts": [],
                "size": [],
            }
        )

        self.foreground_source = ColumnDataSource(
            {"detection_statistic": [], "snr": []}
        )

    def get_layout(self, height, width):
        self.distribution_plot = figure(
            height=height,
            width=int(width * 0.55),
            y_axis_type="log",
            x_axis_label="Detection statistic",
            y_axis_label="Background survival function",
            y_range=(0, 1),  # set dummy values to allow updating later
            tools="box_zoom,reset",
        )
        self.distribution_plot.yaxis.axis_label_text_color = self.bckgd_color

        # add box select tool for selecting ranges
        # of background events to further analyze
        box_select = BoxSelectTool(dimensions="width")
        self.distribution_plot.add_tools(box_select)
        self.distribution_plot.toolbar.active_drag = box_select
        self.bar_source.selected.on_change("indices", self.update_background)

        self.distribution_plot.extra_y_ranges = {"SNR": Range1d(1, 10)}
        axis = LogAxis(
            axis_label="Injected Event SNR",
            axis_label_text_color=self.frgd_color,
            y_range_name="SNR",
        )
        self.distribution_plot.add_layout(axis, "right")

        self.background_plot = figure(
            height=height,
            width=int(width * 0.45),
            title="",
            x_axis_label="GPS Time [s]",
            y_axis_label="Detection statistic",
            tools="box_zoom,reset",
        )

        self.distribution_plot.add_tools(PanTool(), WheelZoomTool())
        self.background_plot.add_tools(PanTool(), WheelZoomTool())

        self.sliders = {}
        for attr in SLIDER_ATTRS:
            # Placeholder ranges that will be set in `update()` once
            # the foreground data is loaded
            slider = RangeSlider(
                start=0,
                end=1,
                value=(0, 1),
                step=1,
                title=attr,
            )
            self.sliders[attr] = slider

        self.update_button = Button(
            label="Update Foreground Event Filter",
            button_type="success",
            width=300,
        )
        self.update_button.on_click(self.update_foreground)

        controls = column(
            *list(self.sliders.values()),
            row(self.update_button),
        )
        plots = row(
            self.distribution_plot, Spacer(width=50), self.background_plot
        )
        self.plot_data()
        return column(controls, plots)

    def update_foreground(self):
        bool_mask = np.ones(
            len(self.foreground_source.data["snr"]), dtype=bool
        )
        for name, slider in self.sliders.items():
            low, high = slider.value
            bool_mask &= (self.foreground_source.data[name] >= low) & (
                self.foreground_source.data[name] <= high
            )
        self.foreground_renderer.view = CDSView(
            filter=BooleanFilter(bool_mask)
        )

    def plot_data(self):
        view = CDSView()
        self.foreground_renderer = self.distribution_plot.scatter(
            x="detection_statistic",
            y="snr",
            fill_color=self.frgd_color,
            line_color=self.frgd_color,
            line_width=0.5,
            fill_alpha=0.2,
            line_alpha=0.4,
            selection_fill_alpha=0.2,
            selection_line_alpha=0.3,
            nonselection_fill_alpha=0.2,
            nonselection_line_alpha=0.3,
            y_range_name="SNR",
            source=self.foreground_source,
            view=view,
        )

        # add hover tool for analyzing additional attributes
        tooltips = [
            ("Injection time", "@{injection_time}{0.000}"),
            ("Shifts", "@shift"),
            ("SNR", "@snr"),
            ("Detection statistic", "@{detection_statistic}"),
            ("Mass 1", "@{mass_1}"),
            ("Mass 2", "@{mass_2}"),
            ("Mass 1 source", "@{mass_1_source}"),
            ("Mass 2 source", "@{mass_2_source}"),
            ("Chirp Mass", "@{chirp_mass}"),
            ("SNR ratio", "@{snr_ratio}"),
        ]

        for ifo in self.page.app.ifos:
            tooltips.append((f"{ifo} SNR", f"@{ifo}_snr"))

        hover = HoverTool(
            tooltips=tooltips,
            renderers=[self.foreground_renderer],
        )
        self.distribution_plot.add_tools(hover)

        tap = TapTool()
        self.foreground_source.selected.on_change(
            "indices", self.inspect_event
        )
        self.distribution_plot.add_tools(tap)

        self.distribution_plot.vbar(
            "center",
            top="top",
            bottom=0.1,
            width="width",
            fill_color=self.bckgd_color,
            line_color="#000000",
            fill_alpha=0.4,
            line_alpha=0.6,
            line_width=0.5,
            selection_fill_alpha=0.6,
            selection_line_alpha=0.8,
            nonselection_fill_alpha=0.2,
            nonselection_line_alpha=0.3,
            source=self.bar_source,
        )

        renderer = self.background_plot.scatter(
            x="x",
            y="detection_statistic",
            fill_color=self.bckgd_color,
            fill_alpha=0.5,
            line_color=self.bckgd_color,
            line_alpha=0.7,
            hover_fill_color=self.bckgd_color,
            hover_fill_alpha=0.7,
            hover_line_color=self.bckgd_color,
            hover_line_alpha=0.9,
            size="size",
            source=self.background_source,
        )

        hover = HoverTool(
            tooltips=[
                ("GPS time", "@{x}{0.000}"),
                ("Detection statistic", "@{detection_statistic}"),
                ("Shifts", "@shifts"),
            ],
            renderers=[renderer],
        )
        self.background_plot.add_tools(hover)

        tap = TapTool()
        self.background_source.selected.on_change(
            "indices", self.inspect_background
        )
        self.background_plot.add_tools(tap)

    def inspect_event(self, attr, old, new):
        if len(new) > 1:
            logging.debug("too many indices")
            return
        if new == old or not new:
            return

        idx = new[0]
        event_time = self.foreground_source.data["injection_time"][idx]
        shift = self.foreground_source.data["shift"][idx]
        snr = self.foreground_source.data["snr"][idx]
        chirp_mass = self.foreground_source.data["chirp_mass"][idx]

        title = "Injected Event: "
        title += f"SNR = {snr:0.1f}, "
        title += f"Chirp Mass = {chirp_mass:0.1f} "

        self.event_inspector.update(
            event_time,
            "foreground",
            shift,
            title,
        )

    def inspect_background(self, attr, old, new):
        if len(new) > 1:
            logging.debug("too many indices")
            return
        if new == old or not new:
            return

        idx = new[0]
        time = self.background_source.data["detection_time"][idx]
        shifts = self.background_source.data["shifts"][idx]

        self.event_inspector.update(
            time, "background", shifts, "Background Event"
        )

    def update_background(self, attr, old, new):
        if len(new) < 2:
            return

        stats = np.array(self.bar_source.data["center"])
        min_ = min([stats[i] for i in new])
        max_ = max([stats[i] for i in new])

        ds = self.background.detection_statistic
        low = np.searchsorted(ds, min_, side="left")
        high = np.searchsorted(ds, max_, side="right")
        n_selected = high - low

        if n_selected == 0:
            self.background_plot.title.text = (
                f"0 events with detection statistic in "
                f"({min_:0.1f}, {max_:0.1f})"
            )
            self.background_source.data = {
                "x": [],
                "detection_time": [],
                "detection_statistic": [],
                "shifts": [],
                "size": [],
            }
            self.background_source.selected.indices = []
            return

        if n_selected > MAX_SCATTER_POINTS:
            rng = np.random.default_rng()
            sample = np.sort(
                rng.choice(n_selected, size=MAX_SCATTER_POINTS, replace=False)
            )
            idx = low + sample
            self.background_plot.title.text = (
                f"showing {MAX_SCATTER_POINTS:,} of {n_selected:,} events "
                f"with detection statistic in ({min_:0.1f}, {max_:0.1f})"
            )
        else:
            idx = np.arange(low, high)
            self.background_plot.title.text = (
                f"{n_selected} events with detection statistic in "
                f"({min_:0.1f}, {max_:0.1f})"
            )

        events = ds[idx]
        times = self.background.detection_time[idx]
        shifts = self.background.shift[idx]

        t0 = times.min()
        self.background_plot.xaxis.axis_label = f"Time from {t0:0.3f} [hours]"

        x = (times - t0) / 3600

        self.background_source.data = {
            "x": x + shifts.sum(axis=-1) / 3600,
            "detection_time": times,
            "detection_statistic": events,
            "shifts": shifts,
            "size": np.full(len(events), 8),
        }
        self.background_source.selected.indices = []

    def update(self, background, foreground):
        self.background = background
        self.foreground = foreground

        n_fg = len(foreground)
        if n_fg > MAX_FOREGROUND_POINTS:
            rng = np.random.default_rng()
            idx = np.sort(
                rng.choice(n_fg, size=MAX_FOREGROUND_POINTS, replace=False)
            )
            fg_note = f" (showing {MAX_FOREGROUND_POINTS:,} of {n_fg:,})"
        else:
            idx = slice(None)
            fg_note = ""

        title = (
            f"{len(self.background)} background events from "
            f"{self.background.Tb / 3600 / 24:0.2f} days worth "
            f"of data; {n_fg} injections overlayed{fg_note}"
        )
        self.foreground_source.data = self.asdict(self.foreground, idx)

        for attr, slider in self.sliders.items():
            values = getattr(self.foreground, attr)
            if not len(values) > 0:
                continue
            low, high = float(np.min(values)), float(np.max(values))
            slider.start = low
            slider.end = high
            slider.value = (low, high)
            slider.step = (high - low) / 100 or 1

        self.distribution_plot.title.text = title

        ds = self.background.detection_statistic
        edges = np.histogram_bin_edges(ds, bins=100)
        top = len(ds) - np.searchsorted(ds, edges[:-1], side="left")
        self.distribution_plot.y_range.start = 0.1
        self.distribution_plot.y_range.end = 2 * top.max() if len(top) else 1

        self.bar_source.data.update(
            center=(edges[:-1] + edges[1:]) / 2,
            top=top,
            width=0.95 * (edges[1:] - edges[:-1]),
        )

        # update snr axis of plot
        # add extra y axis range to show SNR's of events
        self.distribution_plot.extra_y_ranges["SNR"].start = (
            0.5 * self.foreground.snr.min()
        )
        self.distribution_plot.extra_y_ranges["SNR"].end = (
            2 * self.foreground.snr.max()
        )

        # clear the background plot until we select another
        # range of detection characteristics to plot
        self.background_source.data = {
            "x": [],
            "detection_time": [],
            "detection_statistic": [],
            "shifts": [],
            "size": [],
        }
        self.bar_source.selected.indices = []
        self.foreground_source.selected.indices = []
        self.background_source.selected.indices = []

        self.background_plot.title.text = (
            "Select detection characteristic range at left"
        )
        self.background_plot.xaxis.axis_label = "GPS Time [s]"
