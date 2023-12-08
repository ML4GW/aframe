import luigi

from aframe.tasks.data.condor.base import LDGCondorWorkflow


class StaticMemoryWorkflow(LDGCondorWorkflow):
    def append_memory(self, config):
        config.custom_content.append(("request_memory", self.request_memory))


class DynamicMemoryWorklow(LDGCondorWorkflow):
    max_memory = luigi.Parameter(default="7G")

    def append_memory(self, config):
        config.custom_content.append(
            ("+InitialRequestMemory", self.request_memory)
        )
        config.custom_content.append(
            (
                "request_memory",
                f"ifthenelse(isUndefined(MemoryUsage), {self.request_memory}, int(3*MemoryUsage))",  # noqa
            )
        )
        config.custom_content.append(
            (
                "periodic_release",
                "(HoldReasonCode =?= 26 || HoldReasonCode =?= 34) && (JobStatus == 5)",  # noqa
            )
        )
        config.custom_content.append(
            (
                "periodic_remove",
                f"(JobStatus == 1) && MemoryUsage >= {self.max_memory}",
            )
        )
