import os

import law
import luigi
from law.contrib import htcondor

from aframe.tasks.data import DATAFIND_ENV_VARS


class LDGCondorWorkflow(htcondor.HTCondorWorkflow):
    """
    Base class for law workflows that run via condor on LDG
    """

    condor_directory = luigi.Parameter()
    accounting_group_user = luigi.Parameter(default=os.getenv("LIGO_USERNAME"))
    accounting_group = luigi.Parameter(default=os.getenv("LIGO_GROUP"))
    request_disk = luigi.Parameter(default="1024")
    request_memory = luigi.Parameter(default="32678")
    request_cpus = luigi.IntParameter(default=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.htcondor_log_dir.touch()
        self.htcondor_output_directory().touch()

        # update location of where htcondor
        # job files are stored
        # TODO: law PR that makes this configuration
        # easier / more pythonic
        law.config.update(
            {
                "job": {
                    "job_file_dir": self.job_file_dir,
                    "job_file_dir_cleanup": "False",
                    "job_file_dir_mkdtemp": "False",
                }
            }
        )

    @property
    def name(self):
        return self.__class__.__name__.lower()

    @property
    def htcondor_log_dir(self):
        return law.LocalDirectoryTarget(
            os.path.join(self.condor_directory, "logs")
        )

    @property
    def job_file_dir(self):
        return self.htcondor_output_directory().child("jobs", type="d").path

    @property
    def law_config(self):
        path = os.getenv("LAW_CONFIG_FILE", "")
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        return path

    def build_environment(self):
        # set necessary env variables for
        # required for LDG data access
        environment = '"'
        for envvar in DATAFIND_ENV_VARS:
            environment += f"{envvar}={os.getenv(envvar)} "

        # aws endpoint for s3 transfers
        environment += f'AWS_ENDPOINT_URL={os.getenv("AWS_ENDPOINT_URL")} '

        # forward current path and law config
        environment += f'PATH={os.getenv("PATH")} '
        environment += f"LAW_CONFIG_FILE={self.law_config} "
        environment += f"USER={os.getenv('USER')} "

        # forward any env variables that start with AFRAME_
        # that the law config may need to parse
        for envvar, value in os.environ.items():
            if envvar.startswith("AFRAME_"):
                environment += f"{envvar}={value} "
        environment += '"'
        return environment

    def htcondor_output_directory(self):
        return law.LocalDirectoryTarget(self.condor_directory)

    def htcondor_use_local_scheduler(self):
        return True

    def append_memory(self):
        raise NotImplementedError

    def append_logs(self, config):
        for output in ["log", "output", "error"]:
            ext = output[:3]
            config.custom_content.append(
                (
                    output,
                    os.path.join(
                        self.htcondor_log_dir.path,
                        f"{self.name}-$(Cluster).{ext}",
                    ),
                )
            )

    def htcondor_job_config(self, config, job_num, branches):
        environment = self.build_environment()
        config.custom_content.append(("environment", environment))
        config.custom_content.append(("stream_error", "True"))
        config.custom_content.append(("stream_output", "True"))
        config.custom_content.append(
            ("accounting_group", self.accounting_group)
        )
        config.custom_content.append(
            ("accounting_group_user", self.accounting_group_user)
        )
        config.custom_content.append(("request_disk", self.request_disk))
        config.custom_content.append(("request_cpus", self.request_cpus))
        self.append_memory(config)
        self.append_logs(config)
        return config
