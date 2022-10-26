import logging
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

from rich.progress import Progress

from bbhnet.analysis.distributions import ClusterDistribution
from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment
from bbhnet.parallelize import AsyncExecutor


def replace_part(path: Path, part: str, field: str) -> Path:
    parts = path.parts
    idx = parts.index(part)
    p = Path(parts[0])
    parts = parts[1:idx] + (field,) + parts[idx + 1 :]
    return p.joinpath(*parts)


def find_shift_and_foreground(
    background_segment: Segment, shift: str, foreground_field: str
) -> Tuple[Optional[Segment], Optional[Segment]]:
    try:
        shifted = background_segment.make_shift(shift)
    except ValueError:
        logging.info(
            "No shift {} associated with segment {}".format(
                shift, background_segment
            )
        )
        return None, None

    replace = partial(replace_part, part=shifted.field, field=foreground_field)
    fnames = list(map(replace, shifted.fnames))
    try:
        injection = Segment(fnames)
    except ValueError:
        logging.info(
            "No foreground data associated with shift {} "
            "for segment {}".format(shift.name, background_segment)
        )
        return shifted, None

    return shifted, injection


def load_segments(segments: Tuple[Segment]):
    background, foreground = segments
    yb, t = background.load("out")

    if foreground is not None:
        yf, _ = foreground.load("out")
    else:
        yf = None

    return yf, yb, t


def distribution_dict(ifos: Iterable[str], t_clust: float) -> defaultdict:
    def factory():
        bckgrd = ClusterDistribution("integrated", ifos, t_clust)
        frgrd = ClusterDistribution("integrated", ifos, t_clust)

        return {"background": bckgrd, "foreground": frgrd}

    return defaultdict(factory)


class CallbackFactory(Progress):
    def __init__(
        self,
        ifos: Iterable[str],
        t_clust: float,
        max_tb: float,
        write_dir: Path,
        pool: AsyncExecutor,
    ) -> None:
        super().__init__()

        initials = "".join([i[0] for i in ifos])
        self.shift_pattern = re.compile(rf"(?<=[{initials}])[0-9\.]+")
        self.distributions = distribution_dict(ifos, t_clust)

        self.write_dir = write_dir
        self.write_futures = []

        self.pool = pool
        self.main_task_id = self.add_task(
            "[red]Building background", total=max_tb
        )

    @property
    def Tb(self) -> float:
        return self.tasks[self.main_task_id].completed

    def get_cb(
        self, dist: str, norm: Optional[float], shift: str, write_cb: Callable
    ) -> Callable:
        def cb(f):
            t, y, integrated = f.result()

            shifts = self.shift_pattern.findall(shift)
            shifts = list(map(float, shifts))

            d = self.distributions[norm][dist]
            d.fit((integrated, t), shifts, warm_start=True)

            field = f"{dist}-integrated"
            if norm is not None:
                field += f"_norm-seconds={norm}"
            write_dir = self.write_dir / shift / field
            write_dir.mkdir(parents=True, exist_ok=True)

            future = self.pool.submit(
                write_timeseries,
                write_dir,
                prefix="integrated",
                t=t,
                y=y,
                integrated=integrated,
            )
            future.add_done_callback(write_cb)
            self.write_futures.append(future)

        return cb

    def get_task_cb(self, task_id):
        def cb(f):
            self.update(task_id, advance=1)

        return cb

    def get_task_cbs(self, num_loads: int, num_analyze: int, length: float):
        load_task_id = self.add_task(
            f"[cyan]Loading {num_loads} {length}s timeslides",
            total=num_loads,
        )
        analyze_task_id = self.add_task(
            "[yelllow]Integrating timeslides",
            total=num_analyze,
        )
        write_task_id = self.add_task(
            "[green]Writing integrated timeslides",
            total=num_analyze,
        )

        load_cb = self.get_task_cb(load_task_id)
        analyze_cb = self.get_task_cb(analyze_task_id)
        write_cb = self.get_task_cb(write_task_id)
        return load_cb, analyze_cb, write_cb

    def write_distributions(self):
        for norm, dists in self.distributions.items():
            for dist_type, dist in dists.items():
                # don't do in-place ops here or it will affect
                # the dictionary key itself
                fname = dist_type
                if norm is not None:
                    fname = fname + f"_norm-seconds={norm}"
                fname = fname + ".h5"

                dist.write(self.write_dir / fname)
