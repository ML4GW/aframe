from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bokeh.layouts import LayoutDOM
    from vizapp.app import VizApp


class Page:
    def __init__(self, app: "VizApp") -> None:
        self.app = app
        self.initialize_sources()

    def initialize_sources(self) -> None:
        raise NotImplementedError

    def update(self) -> None:
        return

    def get_layout(self) -> "LayoutDOM":
        raise NotImplementedError
