import importlib
import math
import os

import law
import luigi

from aframe.base import AframeSingularityTask
from aframe.tasks.data.timeslide_waveforms import DeployTimeslideWaveforms
from aframe.tasks.infer import InferLocal


def load_prior(path):
    """
    Load prior from python path string
    """
    module_path, prior = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    prior = getattr(module, prior)
    return prior


class SensitiveVolume(AframeSingularityTask):
    """
    Compute and plot the sensitive volume of an aframe analysis
    """

    mass_combos = luigi.ListParameter(
        description="Mass combinations for which to calculate sensitive volume"
    )
    source_prior = luigi.Parameter(
        "Python path to prior to use for waveform generation"
    )
    dt = luigi.FloatParameter(
        default=math.inf,
        description="Time difference to enforce "
        "between injected and recovered events",
    )
    output_dir = luigi.Parameter(
        description="Path to the directory to save the output plots and data"
    )

    @property
    def default_image(self):
        return "plots.sif"

    def requires(self):
        reqs = {}
        reqs["ts"] = DeployTimeslideWaveforms.req(self)
        reqs["infer"] = InferLocal.req(self)
        return reqs

    def output(self):
        path = os.path.join(self.output_dir, "sensitive_volume.h5")
        return law.LocalFileTarget(path)

    def run(self):
        from pathlib import Path

        from plots.main import main

        foreground, background = self.input()["infer"]
        rejected = self.input()["ts"][1].path
        source_prior = load_prior(self.source_prior)
        main(
            Path(background.path),
            Path(foreground.path),
            Path(rejected),
            mass_combos=self.mass_combos,
            source_prior=source_prior,
            dt=self.dt,
            output_dir=Path(self.output_dir),
        )
