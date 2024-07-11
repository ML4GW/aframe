import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from architectures.base import Architecture
from bokeh.layouts import column, row
from bokeh.models import Div, MultiChoice, TabPanel, Tabs
from plots.data import Data
from plots.pages import Analysis, Summary

if TYPE_CHECKING:
    from plots.vetos import VetoParser


class App:
    """
    Bokeh application class, which sets up the layout of pages
    """

    def __init__(
        self,
        data: Data,
        arch: Architecture,
        whitener: torch.nn.Module,
        snapshotter: torch.nn.Module,
        veto_parser: "VetoParser",
    ) -> None:
        self.logger = logging.getLogger("vizapp")
        self.veto_parser = veto_parser
        self.model = arch
        self.whitener = whitener
        self.snapshotter = snapshotter
        self.data = data
        self.background = data._background
        self.foreground = data._foreground

        # initialize all our pages and their constituent plots
        self.pages, tabs = [], []
        for page in [Summary, Analysis]:
            page = page(self)
            self.pages.append(page)

            title = page.__class__.__name__
            tab = TabPanel(child=page.get_layout(), title=title)
            tabs.append(tab)
            page.update()

        # self.veto_selecter = self.get_veto_selecter()
        # self.veto_selecter.on_change("value", self.update_vetos)
        # self.update_vetos(None, None, [])

        # set up a header with a title and the selecter
        title = Div(text="<h1>aframe Performance Dashboard</h1>", width=500)
        header = row(title)  # , self.veto_selecter)

        # generate the final layout
        tabs = Tabs(tabs=tabs)
        self.layout = column(header, tabs)
        self.logger.info("Application ready!")

    def get_veto_selecter(self):
        options = ["CAT1", "CAT2", "CAT3", "GATES"]
        self.vetoes = {}
        for label in options:
            logging.info(f"Calculating veto mask for {label}")
            vetos = self.veto_parser.get_vetoes(label)
            veto_mask = False
            for i, ifo in enumerate(self.data.ifos):
                segments = vetos[ifo]

                _, mask = self.data._background.apply_vetos(segments, i)

                # mark a background event as vetoed
                # if it is vetoed in any if
                veto_mask |= mask

            self.vetoes[label] = veto_mask

        logging.info("Veto masks calculated")
        self.veto_mask = np.zeros_like(mask, dtype=bool)
        return MultiChoice(title="Applied Vetoes", value=[], options=options)

    def update_vetos(self, attr, old, new):
        if not new:
            # no vetoes selected, so mark all background
            # events as not-vetoed
            self.veto_mask = np.zeros_like(self.veto_mask, dtype=bool)
        else:
            # mark a background event as vetoed if any
            # of the currently selected labels veto it
            mask = False
            for label in new:
                mask |= self.vetoes[label]
            self.veto_mask = mask

        # update vetos in our data object
        self.background = self.data._background[~self.veto_mask]
        self.foreground = self.data._foreground[~self.veto_mask]
        # now update all our pages to factor
        # in the vetoed data
        for page in self.pages:
            page.update()

    def __call__(self, doc):
        doc.add_root(self.layout)
