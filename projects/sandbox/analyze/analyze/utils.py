import logging
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

from rich.progress import Progress

from bbhnet.io.timeslides import Segment


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


class AnalysisProgress(Progress):
    def __init__(self, max_tb: Optional[float] = None) -> None:
        super().__init__()

        self.max_tb = max_tb or float("inf")
        self.main_task_id = self.add_task(
            "[red]Building background", total=max_tb
        )

    @property
    def Tb(self) -> float:
        return self.tasks[self.main_task_id].completed

    def get_task_cb(self, task_id):
        def cb(f):
            self.update(task_id, advance=1)

        return cb

    def get_tasks(self, num_loads: int, num_analyze: int, length: float):
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
        return load_task_id, analyze_task_id, write_task_id
