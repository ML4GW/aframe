from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bokeh.layouts import LayoutDOM
    from plots.app import App


class Page:
    def __init__(self, app: "App") -> None:
        self.app = app

    def initialize_sources(self) -> None:
        raise NotImplementedError

    def update(self) -> None:
        return

    def get_layout(self) -> "LayoutDOM":
        raise NotImplementedError
