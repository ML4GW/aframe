import logging
import os
import subprocess
from pathlib import Path
from typing import TypedDict

import law
import luigi
import psutil
from law.contrib import htcondor
from luigi.util import inherits

import aframe.utils as utils
from aframe.base import AframeGPUTask, AframeSandboxTask

INFER_DIR = Path(__file__).parent.parent.parent.parent / "projects" / "infer"


def get_poetry_env(path):
    output = None
    try:
        output = subprocess.check_output(
            ["poetry", "env", "info", "-p"], cwd=path
        )
    except subprocess.CalledProcessError:
        logging.warning("Infer dir is not a valid poetry env")
    else:
        output = output.decode("utf-8").strip()
    return output


class InferRequires(TypedDict):
    data: law.Task
    waveforms: law.Task
    export: law.Task


class InferenceParams(law.Task):
    output_dir = luigi.Parameter()
    inference_sampling_rate = luigi.FloatParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    cluster_window_length = luigi.FloatParameter()
    integration_window_length = luigi.FloatParameter()
    fduration = luigi.FloatParameter()
    Tb = luigi.FloatParameter()
    shifts = luigi.ListParameter()
    sequence_id = luigi.IntParameter()


@inherits(InferenceParams)
class InferLocal(AframeGPUTask):
    server_log = luigi.Parameter()
    output_dir = luigi.Parameter()
    triton_image = luigi.Parameter()

    env_path = get_poetry_env(INFER_DIR)
    sandbox = f"venv::{env_path}"

    @staticmethod
    def get_ip_address() -> str:
        """
        Get the local, cluster-internal IP address.
        Currently not a general function.
        """
        nets = psutil.net_if_addrs()
        return nets["enp1s0f0"][0].address

    def requires(self) -> InferRequires:
        raise NotImplementedError

    @property
    def foreground_output(self):
        return os.path.join(self.output_dir, "foreground.h5")

    @property
    def background_output(self):
        return os.path.join(self.output_dir, "background.h5")

    @property
    def model_repo_dir(self):
        return self.input()["export"].output().path

    @property
    def background_fnames(self):
        return list(self.input()["data"].collection.targets.values())

    @property
    def injection_set_fname(self):
        return self.input()["waveforms"].output().path

    @property
    def segments(self):
        return utils.segments_from_paths(self.background_fnames)

    @property
    def shifts_required(self):
        max_shift = max(self.shifts)
        return utils.get_num_shifts(self.segments, self.Tb, max_shift)

    def output(self):
        return [
            law.LocalFileTarget(self.foreground_output),
            law.LocalFileTarget(self.background_output),
        ]

    def run(self):
        import shutil

        from ledger.events import EventSet, RecoveredInjectionSet

        from hermes.aeriel.serve import serve

        # initialize a triton server contextmanager
        instance = serve(
            model_repo_dir=self.model_repo_dir,
            image=self.triton_image,
            log_file=self.server_log,
            wait=True,
        )
        # enter the context manager, and launch
        # triton client jobs via conder
        with instance:
            outputs = yield Clients.req(
                self,
                image="infer.sif",
                background_fnames=self.background_fnames,
                injection_set_fname=self.injection_set_fname,
                shifts_required=self.shifts_required,
                sequence_id=self.sequence_id,
            )

            background_fnames = [o.path for o in outputs[1]]
            foreground_fnames = [o.path for o in outputs[0]]

            # aggregate results
            background, foreground = EventSet(), RecoveredInjectionSet()
            for b, f in zip(background_fnames, foreground_fnames):
                bckground = EventSet.read(b)
                frground = RecoveredInjectionSet.read(f)

                background.append(bckground)
                foreground.append(frground)

            background.write(self.background_output)
            foreground.write(self.foreground_output)
            shutil.rmtree(self.output_dir / "tmp")


@inherits(InferenceParams)
class Clients(htcondor.HTCondorWorkflow, AframeSandboxTask):
    shifts_required = luigi.IntParameter()
    background_fnames = luigi.ListParameter()
    injection_set_fname = luigi.Parameter()
    sequence_id = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_branch_map(self):
        branch_map, i = {}, 0
        for fname in self.background_fnames:
            for i in range(self.shifts_required):
                seq_id = self.sequence_id + 2 * i
                shifts = [(i + 1) * shift for shift in self.shifts]
            branch_map[i] = (fname, shifts, seq_id)
        return branch_map

    @property
    def tmp_dir(self):
        return os.path.join(self.output_dir, "tmp", f"output-{self.branch}")

    @property
    def foreground_output(self):
        return os.path.join(self.tmp_dir, "foreground.h5")

    @property
    def background_output(self):
        return os.path.join(self.tmp_dir, "background.h5")

    def output(self):
        return [
            law.LocalFileTarget(self.foreground_output),
            law.LocalFileTarget(self.background_output),
        ]

    def run(self):
        from infer import infer

        fname, shifts, seq_id = self.branch_data
        os.makedirs(self.tmp_dir, exist_ok=True)
        infer(
            ip=f"{self.ip}:8001",
            model_name="aframe",
            model_version=-1,
            shifts=shifts,
            background_fname=fname,
            injection_set_fname=self.injection_set_fname,
            batch_size=self.batch_size,
            psd_length=self.psd_length,
            fduration=self.fduration,
            inference_sampling_rate=self.inference_sampling_rate,
            integration_window_length=self.integration_window_length,
            cluster_window_length=self.cluster_window_length,
            sequence_id=seq_id,
        )
