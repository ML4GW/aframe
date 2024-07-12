import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List

import torch
from architectures.base import Architecture
from bokeh.layouts import column, row
from bokeh.models import Div, TabPanel, Tabs
from plots.data import DataManager
from plots.pages import Analysis, Summary

from utils.s3 import open_file

if TYPE_CHECKING:
    from plots.pages import Page


class App:
    """
    Bokeh application class, which sets up the layout of pages
    """

    def __init__(
        self,
        architecture: Architecture,
        weights: str,
        data_dir: Path,
        results_dir: Path,
        ifos: List[str],
        mass_combos: List[tuple],
        source_prior: Callable,
        kernel_length: float,
        psd_length: float,
        highpass: float,
        batch_size: int,
        sample_rate: float,
        inference_sampling_rate: float,
        integration_length: float,
        fduration: float,
        valid_frac: float,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.logger = logging.getLogger("vizapp")

        # set attributes that will be accessible
        # to downstream pages
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.ifos = ifos
        self.mass_combos = mass_combos
        self.source_prior, _ = source_prior()
        self.kernel_length = kernel_length
        self.psd_length = psd_length
        self.highpass = highpass
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.inference_sampling_rate = inference_sampling_rate
        self.integration_length = integration_length
        self.fduration = fduration
        self.valid_frac = valid_frac
        self.device = device
        self.verbose = verbose
        self.architecture = architecture
        self.weights = weights
        self.device = device

        # load in the model weights
        self.model = self.load_model()

        # instantiate object for managing
        # data loading and application of vetos
        self.data_manager = DataManager(results_dir, data_dir)

        # initialize all our pages and their constituent plots
        self.pages, tabs = [], []

        for page in [Summary, Analysis]:
            page: "Page" = page(self)
            self.pages.append(page)

            title = page.__class__.__name__
            tab = TabPanel(child=page.get_layout(), title=title)
            tabs.append(tab)

        self.veto_selecter = self.data_manager.get_veto_selecter()
        self.veto_selecter.on_change("value", self.data_manager.update_vetos)
        self.data_manager.update_vetos(None, None, [])

        # set up a header with a title and the selecter
        title = Div(text="<h1>aframe Performance Dashboard</h1>", width=500)
        header = row(title, self.veto_selecter)

        # generate the final layout
        tabs = Tabs(tabs=tabs)
        self.layout = column(header, tabs)
        self.logger.info("Application ready!")

    def load_model(self):
        with open_file(self.weights, "rb") as f:
            weights = torch.load(f, map_location="cpu")["state_dict"]
            weights = {
                k.strip("model."): v
                for k, v in weights.items()
                if k.startswith("model.")
            }
            self.architecture.load_state_dict(weights)
        return self.architecture.to(self.device)

    def __call__(self, doc):
        doc.add_root(self.layout)
