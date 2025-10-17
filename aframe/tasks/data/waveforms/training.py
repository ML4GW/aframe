from pathlib import Path
import shutil

import law
from luigi.util import inherits

from aframe.config import paths
from aframe.parameters import PathParameter, load_prior
from aframe.targets import s3_or_local
from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.condor.workflows import StaticMemoryWorkflow
from aframe.tasks.data.waveforms.base import DeployTask, WaveformParams


@inherits(WaveformParams)
class DeployTrainingWaveforms(
    AframeDataTask, DeployTask, law.LocalWorkflow, StaticMemoryWorkflow
):
    """
    Generate waveforms for training
    """

    condor_directory = PathParameter(
        description="Directory where condor logs will be saved",
        default=paths().condor_dir / "training_waveforms",
    )

    output_dir = PathParameter(
        description="Directory where merged training waveforms will be saved",
        default=paths().train_waveforms_dir,
    )

    tmp_dir = PathParameter(
        description="Directory where temporary "
        "waveforms will be saved before being merged",
        default=paths().tmp_dir / "training_waveforms",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def output(self):
        return s3_or_local(self.tmp_dir / f"waveforms-{self.branch}.hdf5")

    def run(self):
        from data.waveforms.training import training_waveforms

        self.output_dir.mkdir(exist_ok=True)
        num_signals = self.branch_data
        prior = load_prior(self.prior)
        waveforms = training_waveforms(
            num_signals=num_signals,
            waveform_duration=self.waveform_duration,
            sample_rate=self.sample_rate,
            prior=prior,
            minimum_frequency=self.minimum_frequency,
            reference_frequency=self.reference_frequency,
            waveform_approximant=self.waveform_approximant,
            right_pad=self.right_pad,
        )
        chunks = (min(64, num_signals), waveforms.get_waveforms().shape[-1])
        with self.output().open("w") as f:
            waveforms.write(f, chunks=chunks)


@inherits(DeployTrainingWaveforms)
class TrainingWaveforms(AframeDataTask):
    """
    Launch condorized generation of training waveforms,
    and merge results into a single file
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_file = self.output_dir / "training_waveforms.hdf5"

    def output(self):
        return s3_or_local(self.output_file)

    def requires(self):
        return DeployTrainingWaveforms.req(
            self,
            workflow=self.workflow,
            request_memory=self.request_memory,
            request_disk=self.request_disk,
            request_cpus=self.request_cpus,
        )

    @property
    def targets(self):
        return list(self.input().collection.targets.values())

    @property
    def waveform_files(self):
        return list(map(Path, [targets.path for targets in self.targets]))

    def run(self):
        from ledger.injections import (
            WaveformPolarizationSet,
            waveform_class_factory,
        )

        cls = waveform_class_factory(
            ["cross", "plus"],
            WaveformPolarizationSet,
            "WaveformPolarizationSet",
        )
        with self.output().open("w") as f:
            cls.aggregate(self.waveform_files, f, clean=True)

        # clean up temporary directory
        shutil.rmtree(self.tmp_dir)
