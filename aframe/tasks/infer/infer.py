import os

import law
import luigi
import psutil
from law.contrib import htcondor

from aframe.tasks.data.timeslide_waveforms import MergeTimeslideWaveforms


class TritonServer(luigi.Task):
    image = luigi.Parameter()
    model_repo_dir = luigi.Parameter()
    server_log = luigi.Parameter()
    data_dir = luigi.Parameter()
    output_dir = luigi.Parameter()

    @staticmethod
    def get_ip_address() -> str:
        """
        Get the local, cluster-internal IP address.
        Currently not a general function.
        """
        nets = psutil.net_if_addrs()
        return nets["enp1s0f0"][0].address

    def output(self):
        output = dict(
            name=law.LocalFileTarget(
                os.path.join(self.output_dir, "server_name.txt")
            ),
            ip=law.LocalFileTarget(
                os.path.join(self.output_dir, "server_ip.txt")
            ),
        )
        return output

    def run(self):
        from hermes.aeriel.serve import serve

        # initialize server context, and enter it returning,
        # returning a singularity container
        # python instance pointing to the server.
        # set wait to True to block until the server
        # is live
        instance = serve(
            self.image, self.model_repo_dir, self.server_log, wait=True
        )
        instance = instance.__enter__
        # write the server instance name and for
        # downstream tasks to close the instance
        with self.output()["name"]("w") as f:
            f.write(instance.name)

        with self.output()["ip"]("w") as f:
            f.write(self.get_ip_address())


class TritonClients(htcondor.HTCondorWorkflow):
    ip_address = luigi.Parameter(default="localhost")
    data_dir = luigi.Parameter()
    inference_sampling_rate = luigi.FloatParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    cluster_window_length = luigi.FloatParameter()
    integration_window_length = luigi.FloatParameter()
    fduration = luigi.FloatParameter()

    def create_branch_map(self):
        segments = self.input()["segments"].load().splitlines()[1:]
        branch_map, i = {}, 1
        for row in segments:
            row = row.split("\t")
            start, duration = map(float, row[1::2])
            step = duration if self.max_duration == -1 else self.max_duration
            num_steps = (duration - 1) // step + 1

            for j in range(int(num_steps)):
                segstart = start + j * step
                segdur = min(start + duration - segstart, step)
                branch_map[i] = (segstart, segdur)
                i += 1
        return branch_map

    def workflow_requires(self):
        return MergeTimeslideWaveforms()  # Fetch()

    def requires(self):
        return TritonServer()

    @property
    def tmp_dir(self):
        return os.path.join(self.data_dir, f"tmp-{self.branch}")

    def output(self):
        return law.LocalFileTarget(
            os.path.join(self.data_dir, "client_ips.txt")
        )

    def run(self):
        from infer import infer

        ip = self.input()["ip"].load()
        shifts, background_fname, injection_set_fname = self.branch_data
        infer(
            ip=f"{ip}:8001",
            model_name="aframe",
            model_version=-1,
            shifts=shifts,
            background_fname=background_fname,
            injection_set_fname=injection_set_fname,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            psd_length=self.psd_length,
            fduration=self.fduration,
            inference_sampling_rate=self.inference_sampling_rate,
            integration_window_length=self.integration_window_length,
            cluster_window_length=self.cluster_window_length,
        )


class AggregateInfer(luigi.Task):
    def requires(self):
        return dict(clients=TritonClients(), server=TritonServer())

    def output(self):
        pass

    def run(self):
        # from spython import Client

        pass
