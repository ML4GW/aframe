import law
import luigi


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
    minimum_frequency = luigi.FloatParameter(
        default=20, description="Minimum frequency of the generated signals"
    )
    reference_frequency = luigi.FloatParameter(
        default=50, description="Reference frequency of the generated signals"
    )
    waveform_approximant = luigi.Parameter(
        default="IMRPhenomXPHM",
        description="Approximant to use for waveform generation",
    )
    coalescence_time = luigi.FloatParameter(
        description="Location of the defining point of the signal "
        "within the generated waveform"
    )
    prior = luigi.Parameter(
        "Python path to prior to use for waveform generation"
    )
    prior_args = luigi.DictParameter(
        default={}, 
        description="Optional arguments to pass to the prior callable"
    )
    


class DeployTask(law.Task):
    num_jobs = luigi.IntParameter(
        default=10,
        description="Number of parallel jobs "
        "to split waveform generation amongst",
    )

    def create_branch_map(self):
        # split the number of signals into num_jobs branches
        waveforms_per_branch, remainder = divmod(
            self.num_signals, self.num_jobs
        )
        branches = {i: waveforms_per_branch for i in range(self.num_jobs)}
        branches[0] += remainder
        return branches
