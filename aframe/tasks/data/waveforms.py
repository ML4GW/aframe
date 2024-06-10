import os
import shutil
from pathlib import Path

import law
import luigi
from luigi.util import inherits

from aframe.parameters import PathParameter, load_prior
from aframe.targets import s3_or_local
from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.condor.workflows import StaticMemoryWorkflow
from aframe.tasks.data.fetch import FetchTrain


class WaveformParams(law.Task):
    """
    Parameters for waveform generation tasks
    """

    num_signals = luigi.IntParameter(
        description="Number of signals to generate"
    )
    sample_rate = luigi.FloatParameter(
        description="Sample rate of the generated signals"
    )
    waveform_duration = luigi.FloatParameter(
        description="Duration of the generated signals"
    )
    prior = luigi.Parameter(
        "Python path to prior to use for waveform generation"
    )
    minimum_frequency = luigi.FloatParameter(
        default=20, description="Minimum frequency of the generated signals"
    )
    reference_frequency = luigi.FloatParameter(
        default=50, description="Reference frequency of the generated signals"
    )
    waveform_approximant = luigi.Parameter(
        default="IMRPhenomPv2",
        description="Approximant to use for waveform generation",
    )
    coalescence_time = luigi.FloatParameter(
        description="Location of the defining point of the signal "
        "within the generated waveform"
    )


@inherits(WaveformParams)
class TrainingWaveforms(AframeDataTask):
    """
    Generate waveforms for training
    """

    output_file = PathParameter()

    def output(self):
        return s3_or_local(self.output_file)

    def run(self):
        from data.waveforms.injection import (
            WaveformGenerator,
            convert_to_detector_frame,
            write_waveforms,
        )

        generator = WaveformGenerator(
            self.waveform_duration,
            self.sample_rate,
            self.minimum_frequency,
            self.reference_frequency,
            waveform_approximant=self.waveform_approximant,
            coalescence_time=self.coalescence_time,
        )
        prior = load_prior(self.prior)
        prior, detector_frame_prior = prior()

        samples = prior.sample(self.num_signals)
        if not detector_frame_prior:
            samples = convert_to_detector_frame(samples)
        signals = generator(samples)
        with self.output().open("w") as f:
            write_waveforms(f, signals, samples, generator)


@inherits(WaveformParams)
class DeployValidationWaveforms(
    AframeDataTask,
    law.LocalWorkflow,
    StaticMemoryWorkflow,
):
    """
    Generate waveforms for validation via rejection sampling
    """

    output_dir = PathParameter(
        description="Directory where validation waveforms will be saved"
    )
    tmp_dir = PathParameter(
        description="Directory where temporary validation "
        "waveforms will be saved before being merged",
        default=os.getenv("AFRAME_TMPDIR", f"/local/{os.getenv('USER')}"),
    )
    ifos = luigi.ListParameter(
        description="Interferometers for which waveforms will be generated"
    )
    snr_threshold = luigi.FloatParameter(
        description="SNR threshold for rejection sampling"
    )
    highpass = luigi.FloatParameter(
        description="Frequency of highpass filter in Hz"
    )
    num_jobs = luigi.IntParameter(
        default=10,
        description="Number of parallel jobs "
        "to split waveform generation amongst",
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["psd_segment"] = FetchTrain.req(
            self,
            data_dir=self.output_dir / "background",
            segments_file=self.output_dir / "segments.txt",
        )
        return reqs

    @property
    def branch_tmp_dir(self):
        return self.tmp_dir / f"tmp-{self.branch}"

    def output(self):
        return law.LocalFileTarget(self.branch_tmp_dir / "val_waveforms.hdf5")

    def create_branch_map(self):
        # split the number of signals into num_jobs branches
        waveforms_per_branch, remainder = divmod(
            self.num_signals, self.num_jobs
        )
        branches = {
            i: [waveforms_per_branch, self.psd_segment]
            for i in range(self.num_jobs)
        }
        branches[0][0] += remainder
        return branches

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
        from data.timeslide_waveforms.utils import load_psds
        from data.waveforms.rejection import rejection_sample
        from ledger.injections import WaveformSet, waveform_class_factory

        cls = waveform_class_factory(
            self.ifos,
            WaveformSet,
            "IfoWaveformSet",
        )

        os.makedirs(self.branch_tmp_dir, exist_ok=True)
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

    output_dir = PathParameter(
        description="Directory where merged validation waveforms will be saved"
    )
    condor_directory = PathParameter(
        default=os.path.join(
            os.getenv("AFRAME_CONDOR_DIR", "/tmp/aframe/"),
            "validation_waveforms",
        )
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

        # clean up temporary directories
        for dirname in self.output_dir.glob("tmp-*"):
            shutil.rmtree(dirname)
