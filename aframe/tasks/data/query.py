import law
import luigi

from aframe.tasks.data.base import AframeDataTask


class Query(AframeDataTask):
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    output_file = luigi.Parameter()
    min_duration = luigi.FloatParameter(default=0)
    flags = luigi.ListParameter(default=["DCS-ANALYSIS_READY_C01:1"])
    ifo = luigi.Parameter(default="H1")

    def output(self):
        return law.LocalFileTarget(self.output_file)

    @property
    def command(self):
        args = [
            "--start",
            str(self.start),
            "--end",
            str(self.end),
            "--output-file",
            self.output().path,
        ]
        for flag in self.flags:
            args.append("--flags+=" + self.ifo + ":" + flag)
        if self.min_duration > 0:
            args.append(f"--min_duration={self.min_duration}")
        return self.cli + args
