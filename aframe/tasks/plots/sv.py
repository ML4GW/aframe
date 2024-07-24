import math
import os

import law
import luigi

from aframe.base import AframeSingularityTask
from aframe.config import paths
from aframe.parameters import PathParameter, load_prior
from aframe.tasks.data.waveforms.testing import DeployTestingWaveforms
from aframe.tasks.infer import Infer


class SensitiveVolume(AframeSingularityTask):
    """
    Compute and plot the sensitive volume of an aframe analysis
    """

    ifos = luigi.ListParameter(
        description="List of interferometers used in analysis"
    )
    mass_combos = luigi.ListParameter(
        description="Mass combinations for which to calculate sensitive volume"
    )
    source_prior = luigi.Parameter(
        "Python path to prior used for generating testing waveform injections"
    )
    dt = luigi.FloatParameter(
        default=math.inf,
        description="Time difference to enforce "
        "between injected and recovered events",
    )
    output_dir = PathParameter(
        description="Path to the directory to save the output plots and data",
        default=paths().results_dir / "plots",
    )

    @property
    def default_image(self):
        return "plots.sif"

    def requires(self):
        reqs = {}
        reqs["ts"] = DeployTestingWaveforms.req(self)
        reqs["infer"] = Infer.req(self)
        return reqs

    def output(self):
        path = os.path.join(self.output_dir, "sensitive_volume.h5")
        return law.LocalFileTarget(path)

    def run(self):
        from pathlib import Path

        from plots.legacy.main import main

        foreground = self.input()["infer"]["foreground"]
        background = self.input()["infer"]["background"]
        rejected = self.input()["ts"][1].path
        source_prior = load_prior(self.source_prior)
        main(
            Path(background.path),
            Path(foreground.path),
            Path(rejected),
            self.ifos,
            mass_combos=self.mass_combos,
            source_prior=source_prior,
            dt=self.dt,
            output_dir=Path(self.output_dir),
        )
