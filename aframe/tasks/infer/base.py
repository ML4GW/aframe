from pathlib import Path

import law
import luigi
from luigi.util import inherits

import utils.data as data_utils
from aframe.base import AframeSingularityTask
from aframe.parameters import PathParameter
from aframe.pipelines.config import paths
from aframe.tasks import TestingWaveforms
from aframe.tasks.data.condor.workflows import StaticMemoryWorkflow


class InferParameters(law.Task):
    output_dir = PathParameter()
    ifos = luigi.ListParameter()
    inference_sampling_rate = luigi.FloatParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    cluster_window_length = luigi.FloatParameter()
    integration_window_length = luigi.FloatParameter()
    fduration = luigi.FloatParameter()
    Tb = luigi.FloatParameter()
    shifts = luigi.ListParameter(default=[0, 1])
    sequence_id = luigi.IntParameter()
    model_name = luigi.Parameter()
    model_version = luigi.IntParameter()
    streams_per_gpu = luigi.IntParameter()
    repository_directory = luigi.Parameter(default="")
    rate_per_gpu = luigi.FloatParameter(
        default=100.0, description="Inferences per second per gpu"
    )
    zero_lag = luigi.BoolParameter(default=False)


@inherits(InferParameters)
class InferBase(
    AframeSingularityTask,
    law.LocalWorkflow,
    StaticMemoryWorkflow,
):
    """
    Base class for inference tasks
    """

    @property
    def default_image(self):
        return "infer.sif"

    @property
    def num_clients(self):
        # account for two streams per condor job: background and injection
        return self.streams_per_gpu * self.num_gpus // 2

    @property
    def rate_per_client(self):
        if not self.rate_per_gpu:
            return None
        total_rate = self.num_gpus * self.rate_per_gpu
        return total_rate / self.num_clients

    @property
    def tmp_dir(self):
        return self.output_dir / f"tmp-{self.branch}"

    @property
    def foreground_output(self):
        return self.tmp_dir / "foreground.hdf5"

    @property
    def background_output(self):
        return self.tmp_dir / "background.hdf5"

    @property
    def zero_lag_output(self):
        return self.tmp_dir / "0lag.hdf5"

    @property
    def background_fnames(self):
        return self.input()["data"].collection.targets.values()

    @property
    def injection_set_fname(self):
        return self.input()["waveforms"][0].path

    def workflow_requires(self):
        reqs = {}
        testing_waveforms = TestingWaveforms.req(
            self,
            output_dir=paths().test_datadir,
        )
        fetch = testing_waveforms.requires().workflow_requires()[
            "test_segments"
        ]

        reqs["data"] = fetch
        reqs["waveforms"] = testing_waveforms

        return reqs

    def get_num_shifts(self):
        segments = data_utils.segments_from_paths(self.background_fnames)
        num_shifts = data_utils.get_num_shifts_from_Tb(
            segments, self.Tb, max(self.shifts)
        )
        return num_shifts

    def create_branch_map(self):
        branch_map = {}
        for fname in self.background_fnames:
            start, duration = map(float, Path(fname).stem.split("-")[-2:])
            stop = start + duration
            for i in range(self.num_shifts):
                _shifts = [s * (i + 1) for s in self.shifts]
                # check if segment is long enough to be analyzed
                if data_utils.is_analyzeable_segment(
                    start, stop, _shifts, self.psd_length
                ):
                    branch_map[i] = (fname, _shifts)

            # if its somehow not analyzeable for 0lag then segment
            # length has been set incorrectly, but put this check here anyway
            if self.zero_lag and data_utils.is_analyzeable_segment(
                start, stop, [0] * len(self.shifts), self.psd_length
            ):
                _shifts = [0 for s in self.shifts]
                branch_map[i] = (fname, _shifts)
        return branch_map

    def get_ip_address(self) -> str:
        raise NotImplementedError

    def output(self):
        outputs = {}
        outputs["foreground"] = law.LocalFileTarget(self.foreground_output)
        outputs["background"] = law.LocalFileTarget(self.background_output)
        return outputs

    def run(self):
        from infer.data import Sequence
        from infer.main import infer
        from infer.postprocess import Postprocessor

        from hermes.aeriel.client import InferenceClient

        fname, shifts = self.branch_data
        sequence = Sequence(
            ifos=self.ifos,
            batch_size=self.batch_size,
            inference_sampling_rate=self.inference_sampling_rate,
            rate=self.rate_per_client,
            shifts=shifts,
            background_fname=fname,
            injection_set_fname=self.injection_set_fname,
        )

        postprocessor = Postprocessor(
            integration_window_length=self.integration_window_length,
            inference_sampling_rate=self.inference_sampling_rate,
            cluster_window_lengt=self.cluster_window_length,
            psd_length=self.psd_length,
            fduration=self.fduration,
            t0=sequence.t0,
            shifts=shifts,
        )

        client = InferenceClient(
            address=f"{self.get_ip_address()}:8001",
            model_name=self.model_name,
            model_version=self.model_version,
            callback=sequence,
        )

        infer(client, sequence, postprocessor)
