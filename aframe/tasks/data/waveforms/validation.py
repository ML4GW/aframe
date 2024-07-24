import os
import shutil
from pathlib import Path

import law
import luigi
from luigi.util import inherits

from aframe.config import paths
from aframe.parameters import PathParameter, load_prior
from aframe.targets import s3_or_local
from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.condor.workflows import StaticMemoryWorkflow
from aframe.tasks.data.fetch import FetchTrain
from aframe.tasks.data.waveforms.base import DeployTask, WaveformParams


@inherits(WaveformParams)
class DeployValidationWaveforms(
    AframeDataTask,
    DeployTask,
    law.LocalWorkflow,
    StaticMemoryWorkflow,
):
    """
    Generate waveforms for validation via rejection sampling
    """

    ifos = luigi.ListParameter(
        description="Interferometers for which waveforms will be generated"
    )
    snr_threshold = luigi.FloatParameter(
        description="SNR threshold for rejection sampling"
    )
    highpass = luigi.FloatParameter(
        description="Frequency of highpass filter in Hz"
    )

    output_dir = PathParameter(
        description="Directory where merged training waveforms will be saved",
        default=paths().train_datadir,
    )

    tmp_dir = PathParameter(
        description="Directory where temporary "
        "waveforms will be saved before being merged",
        default=paths().tmp_dir / "val",
    )
    condor_directory = PathParameter(
        description="Directory where condor logs will be saved",
        default=paths().condor_dir / "validation_waveforms",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def output(self):
        return law.LocalFileTarget(
            self.tmp_dir / f"waveforms-{self.branch}.hdf5"
        )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["psd_segment"] = FetchTrain.req(
            self,
            data_dir=self.output_dir / "background",
            segments_file=self.output_dir / "segments.txt",
        )
        return reqs

    def create_branch_map(self):
        branch_map = super().create_branch_map()
        branch_map = {k: (v, self.psd_segment) for k, v in branch_map.items()}
        return branch_map

    @property
    def psd_segment(self):
        """
        Segment to use to calculate psd for rejection sampling
        """
        return list(
            self.workflow_input()["psd_segment"].collection.targets.values()
        )[-1]

    def run(self):
        import io

        import h5py
        from data.waveforms.rejection import rejection_sample
        from data.waveforms.utils import load_psds
        from ledger.injections import WaveformSet, waveform_class_factory

        cls = waveform_class_factory(
            self.ifos,
            WaveformSet,
            "IfoWaveformSet",
        )

        os.makedirs(self.tmp_dir, exist_ok=True)
        num_signals, psd_segment = self.branch_data

        # read in psd
        with psd_segment.open("r") as psd_file:
            psd_file = h5py.File(io.BytesIO(psd_file.read()))
            psds = load_psds(
                psd_file, self.ifos, df=1 / self.waveform_duration
            )

        # load in prior
        prior = load_prior(self.prior)

        # rejection sample waveforms, build
        # waveform set and write to tmp output path
        parameters, _ = rejection_sample(
            num_signals,
            prior,
            self.ifos,
            self.minimum_frequency,
            self.reference_frequency,
            self.sample_rate,
            self.waveform_duration,
            self.waveform_approximant,
            self.coalescence_time,
            self.highpass,
            self.snr_threshold,
            psds,
        )
        waveform_set = cls(**parameters)
        waveform_set.write(self.output().path)


@inherits(DeployValidationWaveforms)
class ValidationWaveforms(AframeDataTask):
    """
    Launch condorized generation of validation waveforms via
    rejection sampling, and merge results into a single file
    """

    condor_directory = PathParameter(
        default=paths().condor_dir / "validation_waveforms"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_file = self.output_dir / "val_waveforms.hdf5"

    def output(self):
        return s3_or_local(self.output_file)

    def requires(self):
        return DeployValidationWaveforms.req(self)

    @property
    def targets(self):
        return list(self.input().collection.targets.values())

    @property
    def waveform_files(self):
        return list(map(Path, [targets.path for targets in self.targets]))

    def run(self):
        from ledger.injections import WaveformSet, waveform_class_factory

        cls = waveform_class_factory(
            self.ifos,
            WaveformSet,
            "WaveformSet",
        )
        with self.output().open("w") as f:
            cls.aggregate(self.waveform_files, f, clean=True)

        # clean up temporary directory
        shutil.rmtree(self.tmp_dir)
