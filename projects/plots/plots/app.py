import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

import torch
import bilby
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Div, MultiChoice, TabPanel, Tabs
from ledger.injections import InjectionParameterSet
from plots.data import Data
from plots.pages.summary import Summary
from architectures.base import Architecture

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology
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
        # veto_parser: "VetoParser",
    ) -> None:
        self.logger = logging.getLogger("vizapp")
        # self.veto_parser = veto_parser
        self.model = arch
        self.whitener = whitener
        self.snapshotter = snapshotter
        self.data = data

        # initialize all our pages and their constituent plots
        self.pages, tabs = [], []
        for page in [Summary]:
            page = page(self)
            self.pages.append(page)

            title = page.__class__.__name__
            tab = TabPanel(child=page.get_layout(), title=title)
            tabs.append(tab)

       

        self.veto_selecter = self.get_veto_selecter()
        self.veto_selecter.on_change(self.update_vetos)
        self.update_vetos(None, None, [])

        # set up a header with a title and the selecter
        title = Div(text="<h1>aframe Performance Dashboard</h1>", width=500)
        # header = row(title, self.veto_selecter)

        # generate the final layout
        tabs = Tabs(tabs=tabs)
        self.layout = column(tabs)
        self.logger.info("Application ready!")

    def get_veto_selecter(self):
        options = ["CAT1", "CAT2", "CAT3", "GATES"]
        self.vetoes = {}
        for label in options:
            vetos = self.veto_parser.get_vetoes(label)
            veto_mask = False
            for ifo in self.ifos:
                segments = vetos[ifo]

                # this will have shape
                # (len(segments), len(self.background))
                mask = segments[:, :1] < self.background.time
                mask &= segments[:, 1:] > self.background.time

                # mark a background event as vetoed
                # if it falls into _any_ of the segments
                veto_mask |= mask.any(axis=0)
            self.vetoes[label] = veto_mask

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

        # now update all our pages to factor
        # in the vetoed data
        for page in self.pages:
            page.update()

    def __call__(self, doc):
        doc.add_root(self.layout)
