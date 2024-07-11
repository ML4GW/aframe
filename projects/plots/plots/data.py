import logging
from pathlib import Path
from typing import List

import bilby
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet


class Data:
    """
    Class for holding all the data we need for the visualization
    """

    def __init__(
        self,
        data_dir: Path,
        results_dir: Path,
        mass_combos: List[tuple],
        source_prior: bilby.core.prior.PriorDict,
        ifos: List[str],
        sample_rate: float,
        kernel_length: float,
        psd_length: float,
        highpass: float,
        batch_size: int,
        inference_sampling_rate: float,
        integration_length: float,
        fduration: float,
        valid_frac: float,
        device: str,
    ):
        # initialize data attributes
        self.ifos = ifos
        self.source_prior = source_prior
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.psd_length = psd_length
        self.highpass = highpass
        self.batch_size = batch_size
        self.inference_sampling_rate = inference_sampling_rate
        self.integration_length = integration_length
        self.fduration = fduration
        self.valid_frac = valid_frac
        self.mass_combos = mass_combos
        self.data_dir = data_dir
        self.device = device

        # load results and data from the run we're visualizing
        infer_dir = results_dir / "infer" / "1year"
        rejected = data_dir / "rejected-parameters.hdf5"
        self.response_set = data_dir / "waveforms.hdf5"

        logging.info("Reading in data")
        # load in the background and foreground events;
        # use "_" for background pre veto
        self._background = EventSet.read(infer_dir / "background.hdf5")
        self._foreground = RecoveredInjectionSet.read(
            infer_dir / "foreground.hdf5"
        )

        self.rejected_params = InjectionParameterSet.read(rejected)
        logging.info("Data loaded")

        # move injection masses to source frame
        for obj in [self._foreground, self.rejected_params]:
            for i in range(2):
                attr = f"mass_{i + 1}"
                value = getattr(obj, attr)
                setattr(obj, attr, value / (1 + obj.redshift))

    @property
    def start(self):
        return self._background.detection_time.min()

    @property
    def stop(self):
        return self._background.detection_time.max()

    @property
    def kernel_size(self):
        return int(self.kernel_length * self.sample_rate)
