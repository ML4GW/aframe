from dataclasses import dataclass

from plots.core.sv import SensitiveVolumePlot, SensitiveVolumeResult


@dataclass(frozen=True)
class PlotType:
    """One registered plot type.

    Args:
        name (str): The name of the plot type.
        result_cls (type):
            The data class with the `read`/`write` methods for the plot's data.
        plot_cls (type): The bokeh renderer class for the plot.
    """

    name: str
    result_cls: type
    plot_cls: type


PLOTS: dict[str, PlotType] = {
    "sensitive_volume": PlotType(
        name="sensitive_volume",
        result_cls=SensitiveVolumeResult,
        plot_cls=SensitiveVolumePlot,
    ),
}
