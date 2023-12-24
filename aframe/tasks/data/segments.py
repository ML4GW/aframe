import law
import luigi

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
        return law.LocalFileTarget(self.segments_file)

    @property
    def flags(self):
        flags = []
        for ifo in self.ifos:
            flags.append(f"{ifo}:{self.flag}")
        return flags

    def run(self):
        from data.segments.segments import DataQualityDict

        segments = DataQualityDict.query_segments(
            self.flags,
            self.start,
            self.end,
            self.min_duration,
        )
        with self.output().open("w") as f:
            segments.write(f, format="segwizard")
