import os
import shutil
from pathlib import Path

import law
from luigi.util import inherits

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        from data.waveforms.injection import convert_to_detector_frame
        from ledger.injections import (
            IntrinsicParameterSet,
            IntrinsicWaveformSet,
        )

        os.makedirs(self.tmp_dir, exist_ok=True)
        num_signals = self.branch_data
        prior = load_prior(self.prior)
        prior, detector_frame_prior = prior()

        samples = prior.sample(num_signals)
        if not detector_frame_prior:
            samples = convert_to_detector_frame(samples)

        for key in ["ra", "dec"]:
            samples.pop(key)

        params = IntrinsicParameterSet(**samples)
        waveforms = IntrinsicWaveformSet.from_parameters(
            params,
            self.minimum_frequency,
            self.reference_frequency,
            self.sample_rate,
            self.waveform_duration,
            self.waveform_approximant,
            self.coalescence_time,
        )
        chunks = (min(64, num_signals), waveforms.get_waveforms().shape[-1])
        waveforms.write(self.output().path, chunks=chunks)


@inherits(DeployTrainingWaveforms)
class TrainingWaveforms(AframeDataTask):
    """
    Launch condorized generation of validation waveforms via
    rejection sampling, and merge results into a single file
    """

    output_dir = PathParameter(
        description="Directory where merged training waveforms will be saved"
    )
    condor_directory = PathParameter(
        default=os.path.join(
            os.getenv("AFRAME_CONDOR_DIR", "/tmp/aframe/"),
            "train_waveforms",
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_file = self.output_dir / "train_waveforms.hdf5"

    def output(self):
        return s3_or_local(self.output_file)

    def requires(self):
        return DeployTrainingWaveforms.req(self)

    @property
    def targets(self):
        return list(self.input().collection.targets.values())

    @property
    def waveform_files(self):
        return list(map(Path, [targets.path for targets in self.targets]))

    def run(self):
        from ledger.injections import IntrinsicWaveformSet

        with self.output().open("w") as f:
            IntrinsicWaveformSet.aggregate(self.waveform_files, f, clean=True)

        # clean up temporary directory
        shutil.rmtree(self.tmp_dir)
