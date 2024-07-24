import os
from pathlib import Path

import law
import luigi
from luigi.util import inherits

import utils.data as data_utils
from aframe.base import AframeSingularityTask
from aframe.config import paths
from aframe.parameters import PathParameter
from aframe.tasks import ExportLocal, TestingWaveforms
from aframe.tasks.data.condor.workflows import StaticMemoryWorkflow


class InferParameters(law.Task):
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
    rate_per_gpu = luigi.FloatParameter(
        default=100.0, description="Inferences per second per gpu"
    )
    clean = luigi.BoolParameter(default=True)
    zero_lag = luigi.BoolParameter(default=False)
    output_dir = PathParameter(default=paths().results_dir)


@inherits(InferParameters)
class InferBase(
    AframeSingularityTask,
    law.LocalWorkflow,
    StaticMemoryWorkflow,
):
    """
    Base class for inference tasks
    """

    condor_directory = PathParameter(default=paths().condor_dir / "infer")

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
        return self.output_dir / "tmp" / f"tmp-{self.branch}"

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
        return self.workflow_input()["data"].collection.targets.values()

    @property
    def injection_set_fname(self):
        return self.workflow_input()["waveforms"][0].path

    def workflow_requires(self):
        reqs = {}
        reqs["model_repository"] = ExportLocal.req(self)
        testing_waveforms = TestingWaveforms.req(self)
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

    @law.dynamic_workflow_condition
    def workflow_condition(self) -> bool:
        return self.workflow_input()["data"].collection.exists()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branch_map = {}
        for fname in self.background_fnames:
            fname = Path(fname.path)
            start, duration = map(float, fname.stem.split("-")[-2:])
            stop = start + duration
            for i in range(self.get_num_shifts()):
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

    @workflow_condition.output
    def output(self):
        outputs = {}
        outputs["foreground"] = law.LocalFileTarget(self.foreground_output)
        outputs["background"] = law.LocalFileTarget(self.background_output)
        return outputs

    def htcondor_job_config(self, config, job_num, branches):
        config = super().htcondor_job_config(config, job_num, branches)
        config.custom_content.append(("max_materialize", self.num_clients))
        return config

    def run(self):
        from infer.data import Sequence
        from infer.main import infer
        from infer.postprocess import Postprocessor

        from hermes.aeriel.client import InferenceClient

        ip = os.getenv("AFRAME_TRITON_IP")
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
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
            cluster_window_length=self.cluster_window_length,
            psd_length=self.psd_length,
            fduration=self.fduration,
            t0=sequence.t0,
            shifts=shifts,
        )

        client = InferenceClient(
            address=f"{ip}:8001",
            model_name=self.model_name,
            model_version=self.model_version,
            callback=sequence,
        )

        with client:
            background, foreground = infer(client, sequence, postprocessor)

        background.write(self.background_output)
        foreground.write(self.foreground_output)
