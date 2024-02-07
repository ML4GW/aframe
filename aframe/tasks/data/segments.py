import luigi

from aframe.targets import s3_or_local
from aframe.tasks.data.base import AframeDataTask


class Query(AframeDataTask):
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    segments_file = luigi.Parameter()
    flag = luigi.Parameter()
    ifos = luigi.ListParameter()
    min_duration = luigi.FloatParameter(default=0)

    retry_count = 3

    def output(self):
        return s3_or_local(self.segments_file, format="txt")

    def get_flags(self):
        if self.flag == "DATA":
            flags = [f"{ifo}_DATA" for ifo in self.ifos]  # open data flags
        else:
            flags = [f"{ifo}:{self.flag}" for ifo in self.ifos]
        return flags

    def run(self):
        from data.segments.segments import DataQualityDict

        flags = self.get_flags()
        segments = DataQualityDict.query_segments(
            flags,
            self.start,
            self.end,
            self.min_duration,
        )
        with self.output().open("w") as f:
            segments.write(f, format="segwizard")
