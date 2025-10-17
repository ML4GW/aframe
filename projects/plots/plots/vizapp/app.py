import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
from bokeh.layouts import column, row
from bokeh.models import Div, TabPanel, Tabs

from plots.vetos import VETO_CATEGORIES
from plots.vizapp.data import DataManager
from plots.vizapp.pages import Analysis, Summary
from utils.logging import configure_logging
from utils.s3 import open_file

if TYPE_CHECKING:
    from plots.pages import Page


class App:
    """
    Bokeh application class, which sets up the layout of pages
    """

    def __init__(
        self,
        weights: str,
        background_dir: Path,
        waveforms_dir: Path,
        results_dir: Path,
        ifos: List[str],
        mass_combos: List[tuple],
        source_prior: Callable,
        kernel_length: float,
        psd_length: float,
        highpass: float,
        lowpass: float,
        batch_size: int,
        sample_rate: float,
        inference_sampling_rate: float,
        integration_length: float,
        fduration: float,
        valid_frac: float,
        fftlength: float,
        device: str = "cpu",
        vetos: Optional[VETO_CATEGORIES] = None,
        verbose: bool = False,
    ) -> None:
        configure_logging(verbose=verbose)
        self.logger = logging.getLogger("vizapp")

        # set attributes that will be accessible
        # to downstream pages
        self.background_dir = background_dir
        self.waveforms_dir = waveforms_dir
        self.results_dir = results_dir
        self.ifos = ifos
        self.mass_combos = mass_combos
        self.source_prior, _ = source_prior()
        self.kernel_length = kernel_length
        self.psd_length = psd_length
        self.highpass = highpass
        self.lowpass = lowpass
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.inference_sampling_rate = inference_sampling_rate
        self.integration_length = integration_length
        self.fduration = fduration
        self.valid_frac = valid_frac
        self.device = device
        self.verbose = verbose
        self.weights = weights
        self.device = device
        self.fftlength = fftlength

        # load in the model weights
        self.model = self.load_model()

        # instantiate object for managing
        # data loading and application of vetos
        self.data_manager = DataManager(
            results_dir, waveforms_dir, ifos, vetos
        )

        # initialize all our pages and their constituent plots
        self.pages: list["Page"] = []
        tabs = []

        for page in [Summary, Analysis]:
            page: "Page" = page(self)
            self.pages.append(page)

            title = page.__class__.__name__
            tab = TabPanel(child=page.get_layout(), title=title)
            tabs.append(tab)

        self.veto_selecter = self.data_manager.get_veto_selecter()
        self.veto_selecter.on_change("value", self.update)
        self.update(None, None, [])

        # set up a header with a title and the selecter
        title = Div(text="<h1>aframe Performance Dashboard</h1>", width=500)
        header = row(title, self.veto_selecter)

        # generate the final layout
        tabs = Tabs(tabs=tabs)
        self.layout = column(header, tabs)
        self.logger.info("Application ready!")

    def load_model(self):
        with open_file(self.weights, "rb") as f:
            model = torch.jit.load(f)
        return model.to(self.device)

    def update(self, attr, old, new):
        # update the vetos
        background, foreground = self.data_manager.update_vetos(attr, old, new)

        # update pages with latest background and foreground
        for page in self.pages:
            page.update(background, foreground)

    def __call__(self, doc):
        doc.add_root(self.layout)
