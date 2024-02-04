from aframe.pipelines.base import TrainDatagen
from aframe.tasks.train import TuneRemote as _TuneRemote


class TuneRemote(_TuneRemote):
    def requires(self):
        yield TrainDatagen.req(self)
