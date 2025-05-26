import json
import os
import warnings
from pathlib import Path

import h5py
import law
import luigi
import numpy as np
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
    zero_lag = luigi.BoolParameter(default="true")
    return_timeseries = luigi.BoolParameter(default="false")
    output_dir = PathParameter(default=paths().results_dir)
    train_task = luigi.TaskParameter()


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # law workflow attribute that sets
        # the condor concurrency;
        # multiply by two since there seems to be
        # some latency in laws job submission
        self.parallel_jobs = int(self.num_clients * 2)

        # Threshold roughly based on the amount of data produced by one
        # year of timeslides at 4 Hz
        if (
            self.return_timeseries
            and self.Tb * self.inference_sampling_rate > 3e8
        ):
            warnings.warn(
                "return_timeseries is set to 'True', and based on the given "
                f"Tb {self.Tb} and inference sampling rate "
                f"{self.inference_sampling_rate}, this will generate a large "
                "amount of data, and the aggregation process will be slow. It "
                "is not recommended to save the output timeseries for long "
                "timeslides or high inference rates",
                stacklevel=2.0,
            )

    @property
    def default_image(self):
        return "infer.sif"

    @property
    def num_clients(self):
        # account for two streams per condor job: background and injection
        return self.streams_per_gpu * self.num_gpus // 2

    @property
    def rate_per_client(self):
        # calculate rate of inference
        # submissions per client based on the rate per gpu
        # and the number of gpus in the deployment
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
    def timeseries_output(self):
        return self.tmp_dir / "timeseries.hdf5"

    @property
    def metadata_output(self):
        return self.tmp_dir / "metadata.json"

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
        # calculate the number of shifts required
        # to accumulate the requested background,
        # given the duration of the background segments
        segments = data_utils.segments_from_paths(self.background_fnames)
        num_shifts = data_utils.get_num_shifts_from_Tb(
            segments, self.Tb, max(self.shifts)
        )
        return num_shifts

    @law.dynamic_workflow_condition(cache_met_condition=True)
    def workflow_condition(self) -> bool:
        return self.workflow_input()["data"].collection.exists()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        # create the individual fname shift
        # combinations that represent individual
        # condor inference jobs to be submitted
        branch_map = {}
        num_shifts = self.get_num_shifts()
        counter = 0
        for fname in self.background_fnames:
            fname = Path(fname.path)
            start, duration = map(float, fname.stem.split("-")[-2:])
            stop = start + duration

            if self.zero_lag:
                # check if segment is long enough to be analyzed
                if data_utils.is_analyzeable_segment(
                    start, stop, [0] * len(self.shifts), self.psd_length
                ):
                    _shifts = [0 for s in self.shifts]
                    branch_map[counter] = (fname, _shifts)
                    counter += 1

            if num_shifts > 0:
                for i in range(num_shifts):
                    _shifts = [s * (i + 1) for s in self.shifts]
                    # check if segment is long enough to be analyzed
                    if data_utils.is_analyzeable_segment(
                        start, stop, _shifts, self.psd_length
                    ):
                        # unique identifier for mapping to branch map
                        branch_map[counter] = (fname, _shifts)
                        counter += 1

        return branch_map

    def get_ip_address(self) -> str:
        raise NotImplementedError

    @workflow_condition.output
    def output(self):
        outputs = {}
        outputs["foreground"] = law.LocalFileTarget(self.foreground_output)
        outputs["background"] = law.LocalFileTarget(self.background_output)
        outputs["metadata"] = law.LocalFileTarget(self.metadata_output)
        if self.return_timeseries:
            outputs["timeseries"] = law.LocalFileTarget(self.timeseries_output)
        return outputs

    def run(self):
        from hermes.aeriel.client import InferenceClient

        from infer.data import Sequence
        from infer.main import infer
        from infer.postprocess import Postprocessor

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
            outputs = infer(
                client, sequence, postprocessor, self.return_timeseries
            )
        if self.return_timeseries:
            background, foreground, background_ts, foreground_ts = outputs
            with h5py.File(self.timeseries_output, "w") as f:
                f.attrs["t0"] = postprocessor.t0
                f.attrs["shifts"] = postprocessor.shifts
                f.create_dataset("background", data=background_ts)
                if foreground_ts is None:
                    foreground_ts = np.zeros(0)
                f.create_dataset("foreground", data=foreground_ts)
        else:
            background, foreground = outputs

        background.write(self.background_output)
        foreground.write(self.foreground_output)

        # Create metadata files to store key information.
        # Although this information is also contained in
        # the hdf5 files that get created, reading these
        # json files seems to be O(1000) times faster.
        metadata = {
            "background_length": len(background),
            "foreground_length": len(foreground),
            "shifts": shifts,
        }
        with open(self.metadata_output, "w") as f:
            json.dump(metadata, f)
