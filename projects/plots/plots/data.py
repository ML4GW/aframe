from pathlib import Path
from typing import List

from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
import bilby

class Data:
    """
    Class for holding all the data we need for the visualization
    """

    def __init__(
        self,
        base_dir: Path,
        data_dir: Path,
        mass_combos: List[tuple],
        source_prior: bilby.core.prior.PriorDict,
        ifos: List[str],
        sample_rate: float,
        kernel_length: float,
        psd_length: float,
        highpass: float,
        batch_size: int,
        inference_sampling_rate: float,
        fduration: float,
        valid_frac: float,
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
        self.fduration = fduration
        self.valid_frac = valid_frac
        self.mass_combos = mass_combos


        # load results and data from the run we're visualizing
        infer_dir = base_dir / "results" / "infer"
        rejected = data_dir / "test" / "rejected-parameters.hdf5"
        self.background = EventSet.read(infer_dir / "background.hdf5")
        self.foreground = RecoveredInjectionSet.read(
            infer_dir / "foreground.hdf5"
        )
        self.rejected_params = InjectionParameterSet.read(rejected)

        # move injection masses to source frame
        for obj in [self.foreground, self.rejected_params]:
            for i in range(2):
                attr = f"mass_{i + 1}"
                value = getattr(obj, attr)
                setattr(obj, attr, value / (1 + obj.redshift))

    @property
    def start(self):
        return self.background.detection_time.min()

    @property
    def stop(self):
        return self.background.detection_time.max()

    @property
    def kernel_size(self):
        return int(self.kernel_length * self.sample_rate)

