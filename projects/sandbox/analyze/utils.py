from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, Union

import numpy as np

from bbhnet.io.timeslides import Segment

if TYPE_CHECKING:
    from bbhnet.analysis.distributions.distribution import Distribution

MAYBE_SEGMENTS = Union[Segment, Iterable[Segment]]


# TODO: move these functions to library?
def load_segments(segments: MAYBE_SEGMENTS, dataset: str):
    """
    Quick utility function which just wraps a Segment's
    `load` method so that we can execute it in a process
    pool since methods aren't picklable.
    """
    # allow iterable of segments
    # for case where we want to load
    # background and injection together

    if isinstance(segments, Segment):
        segments.load(dataset)

    else:
        for segment in segments:
            segment.load(dataset)
    return segments


# TODO: Should characterize_events just be a function
# and not a method for the Distirbution class?
def characterize_events(
    background: "Distribution",
    segment: Union["Segment", Tuple[np.ndarray, np.ndarray]],
    event_times: Union[float, Iterable[float]],
    window_length: float = 1,
    metric: str = "far",
):
    fars, latencies = background.characterize_events(
        segment,
        event_times,
        window_length,
        metric,
    )

    return fars, latencies, event_times


def get_write_dir(
    write_dir: Path,
    shift: Union[str, Segment],
    label: str,
    norm: Optional[float] = None,
) -> Path:
    """
    Quick utility function for getting the name of the directory
    to which to save the outputs from an analysis using a particular
    time-shift/norm-seconds combination
    """

    if isinstance(shift, Segment):
        shift = shift.shift

    if norm is not None:
        write_dir = write_dir / shift / f"{label}-norm-seconds.{norm}"
    else:
        write_dir = write_dir / shift / f"{label}"
    write_dir.mkdir(parents=True, exist_ok=True)
    return write_dir
