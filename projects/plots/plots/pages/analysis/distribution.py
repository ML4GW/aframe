import logging

import numpy as np
from bokeh.layouts import row
from bokeh.models import (
    BoxSelectTool,
    ColumnDataSource,
    HoverTool,
    LogAxis,
    Range1d,
    TapTool,
)
from bokeh.plotting import figure
from plots import palette

FORE_ATTRS = [
    "shift",
    "mass_1",
    "mass_2",
    "snr",
    "detection_statistic",
    "shift",
    "injection_time",
]
BACK_ATTRS = ["detection_statistic", "detection_time"]


class DistributionPlot:
    def __init__(self, page, event_inspector) -> None:
        self.page = page
        self.event_inspector = event_inspector
        self.bckgd_color = palette[4]
        self.frgd_color = palette[2]

    def asdict(self, background, foreground):
        background = {attr: getattr(background, attr) for attr in BACK_ATTRS}
        foreground = {attr: getattr(foreground, attr) for attr in FORE_ATTRS}
        return background, foreground

    def initialize_sources(self):
        self.bar_source = ColumnDataSource(dict(center=[], top=[], width=[]))
        self.background_source = ColumnDataSource(
            dict(
                x=[],
                detection_time=[],
                detection_statistic=[],
                color=[],
                label=[],
                count=[],
                shift=[],
                size=[],
            )
        )

        self.foreground_source = ColumnDataSource(
            dict(size=[], detection_statistic=[], snr=[])
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
        # self.bar_source.selected.on_change("indices", self.update_background)

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

        hover = HoverTool(
            tooltips=[
                ("GPS time", "@{time}{0.000}"),
                ("Detection statistic", "@{detection_statistic}"),
                ("Count", "@count"),
            ]
        )
        self.background_plot.add_tools(hover)
        self.background_plot.legend.click_policy = "hide"

        self.plot_data()
        return row(self.distribution_plot, self.background_plot)

    def plot_data(self):
        injection_renderer = self.distribution_plot.scatter(
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
        )

        # add hover tool for analyzing additional attributes
        hover = HoverTool(
            tooltips=[
                ("Hanford GPS time", "@{time}{0.000}"),
                ("Shifts", "@shift"),
                ("SNR", "@snr"),
                ("Detection statistic", "@{detection_statistic}"),
                ("Mass 1", "@{mass_1}"),
                ("Mass 2", "@{mass_2}"),
                ("Mass 1 source", "@{mass_1_source}"),
                ("Mass 2 source", "@{mass_2_source}"),
                ("Chirp Mass", "@{chirp_mass}"),
            ],
            renderers=[injection_renderer],
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

        self.background_plot.scatter(
            "x",
            "detection_statistic",
            fill_color="color",
            fill_alpha=0.5,
            line_color="color",
            line_alpha=0.7,
            hover_fill_color="color",
            hover_fill_alpha=0.7,
            hover_line_color="color",
            hover_line_alpha=0.9,
            size="size",
            legend_group="label",
            source=self.background_source,
        )
        tap = TapTool()
        self.background_source.selected.on_change(
            "indices", self.inspect_background
        )
        self.background_plot.add_tools(tap)

    def inspect_event(self, attr, old, new):
        if len(new) > 1:
            logging.debug("too many indices")
            return
        if new == old:
            return

        idx = new[0]
        event_time = self.foreground_source.data["injection_time"][idx]
        shift = self.foreground_source.data["shift"][idx]
        snr = self.foreground_source.data["snr"][idx]
        # chirp_mass = self.foreground_source.data["chirp_mass"][idx]

        title = "Injected Event: "
        title += f"SNR = {snr:0.1f}, "
        # title += f"Chirp Mass = {chirp_mass:0.1f}"

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
        if new == old:
            return

        idx = new[0]
        event_time = self.background_source.data["detection_time"][idx]
        label = self.background_source.data["label"][idx]
        shift = self.background_source.data["shift"][idx]
        if label == "Livingston":
            event_time -= shift

        self.event_inspector.update(
            event_time, "background", [0.0, shift], "Background Event"
        )

    def update(self):
        self.background = self.page.app.data.background
        self.foreground = self.page.app.data.foreground

        title = (
            "{} background events from {:0.2f} "
            "days worth of data; {} injections overlayed"
        ).format(
            len(self.background),
            self.background.Tb / 3600 / 24,
            len(self.foreground),
        )
        background_dict, foreground_dict = self.asdict(
            self.background, self.foreground
        )

        self.background_source.data = background_dict
        self.foreground_source.data = foreground_dict

        self.distribution_plot.title.text = title

        # update bar plot
        hist, bins = np.histogram(
            self.background.detection_statistic, bins=100
        )
        hist = np.cumsum(hist[::-1])[::-1]
        self.distribution_plot.y_range.start = 0.1
        self.distribution_plot.y_range.end = 2 * hist.max()

        self.bar_source.data.update(
            center=(bins[:-1] + bins[1:]) / 2,
            top=hist,
            width=0.95 * (bins[1:] - bins[:-1]),
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
        self.background_source.data.update(
            dict(
                x=[],
                detection_time=[],
                detection_statistic=[],
                color=[],
                label=[],
                count=[],
                shift=[],
                size=[],
            )
        )
        self.bar_source.selected.indices = []
        self.foreground_source.selected.indices = []
        self.background_source.selected.indices = []

        self.background_plot.title.text = (
            "Select detection characteristic range at left"
        )
        self.background_plot.xaxis.axis_label = "GPS Time [s]"
