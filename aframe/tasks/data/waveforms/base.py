import os

import law
import luigi

from aframe.parameters import PathParameter


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


class DeployTask(law.Task):
    """
    Common parameters for training and validation waveform generation
    that utililze condor parallelization
    """

    output_dir = PathParameter(
        description="Directory where merged waveforms will be saved"
    )

    tmp_dir = PathParameter(
        description="Directory where temporary "
        "waveforms will be saved before being merged",
        default=os.getenv("AFRAME_TMPDIR", f"/local/{os.getenv('USER')}"),
    )

    num_jobs = luigi.IntParameter(
        default=10,
        description="Number of parallel jobs "
        "to split waveform generation amongst",
    )

    def output(self):
        return law.LocalFileTarget(
            self.tmp_dir / f"waveforms-{self.branch}.hdf5"
        )

    def create_branch_map(self):
        # split the number of signals into num_jobs branches
        waveforms_per_branch, remainder = divmod(
            self.num_signals, self.num_jobs
        )
        branches = {i: waveforms_per_branch for i in range(self.num_jobs)}
        branches[0] += remainder
        return branches
