import law
from luigi.util import inherits

from aframe.config import paths
from aframe.parameters import PathParameter, load_prior
from aframe.targets import s3_or_local
from aframe.tasks.data.base import AframeDataTask
from aframe.tasks.data.condor.workflows import StaticMemoryWorkflow
from aframe.tasks.data.waveforms.base import DeployTask, WaveformParams


@inherits(WaveformParams)
class TrainingWaveforms(
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
        default=paths().train_datadir / "training_waveforms",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def output(self):
        return s3_or_local(self.output_dir / f"waveforms-{self.branch}.hdf5")

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
            prior_args=self.prior_args,
            minimum_frequency=self.minimum_frequency,
            reference_frequency=self.reference_frequency,
            waveform_approximant=self.waveform_approximant,
            coalescence_time=self.coalescence_time,
        )
        chunks = (min(64, num_signals), waveforms.get_waveforms().shape[-1])
        with self.output().open("w") as f:
            waveforms.write(f, chunks=chunks)
