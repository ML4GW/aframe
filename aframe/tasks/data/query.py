import law
import luigi

from aframe.base import logger
from aframe.tasks.data.base import AframeDataTask


class Query(AframeDataTask):
    start = luigi.FloatParameter()
    end = luigi.FloatParameter()
    segments_file = luigi.Parameter()
    min_duration = luigi.FloatParameter(default=0)
    flag = luigi.Parameter()
    ifos = luigi.ListParameter()

    retry_count = 3

    def output(self):
        return law.LocalFileTarget(self.segments_file)

    def get_args(self):
        args = [
            "query",
            "--start",
            str(self.start),
            "--end",
            str(self.end),
            "--output_file",
            self.output().path,
        ]
        for ifo in self.ifos:
            args.append("--flags+=" + ifo + ":" + self.flag)
        if self.min_duration > 0:
            args.append(f"--min_duration={self.min_duration}")

        return args

    def run(self):
        logger.debug(f"Running with args: {' '.join(self.get_args())}")
        from data.cli import main

        logger.debug(f"Running with args: {' '.join(self.get_args())}")
        main(args=self.get_args())
