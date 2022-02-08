import logging
import os
import sys
from pathlib import Path
from typing import Optional

from analyze_o2.utils import build_background
from hermes.typeo import typeo

from bbhnet.io import filter_and_sort_files, fname_re

event_times = [1186302519.8, 1186741861.5, 1187058327.1, 1187529256.5]
event_names = ["GW170809", "GW170814", "GW170818", "GW170823"]


@typeo
def main(
    data_dir: Path,
    write_dir: Path,
    analysis_period: float,
    num_bins: int = 10000,
    window_length: float = 1.0,
    norm_seconds: Optional[float] = None,
    max_tb: Optional[float] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
):
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    if log_file is not None:
        handler = logging.FileHandler(filename=log_file, mode="w")
        logging.getLogger().addHandler(handler)

    data_dir = Path(data_dir)
    groups, current_group, current_t0, lengths = [], [], [], []
    events, current_event = [], None
    last_t0, last_length = None, None

    runs = sorted(map(int, os.listdir(data_dir / "dt-0.0")))
    for run in runs:
        run_dir = data_dir / "dt-0.0" / str(run) / "out"
        for fname in filter_and_sort_files(run_dir):
            t0 = int(fname_re.search(fname).group("t0"))
            length = int(fname_re.search(fname).group("length"))

            fname = Path(str(run)) / "out" / fname
            try:
                new_group = t0 > (last_t0 + last_length)
            except TypeError:
                current_t0 = t0
            else:
                if new_group:
                    groups.append(current_group)
                    events.append(current_event)
                    lengths.append(t0 + length - current_t0)

                    current_group = []
                    current_event = None
                    current_t0 = t0

            current_group.append(fname)
            last_t0 = t0
            last_length = length

            # check to see if there's an event contained in this file
            if current_event is None:
                for event_time, event_name in zip(event_times, event_names):
                    if t0 < event_time < (t0 + length):
                        current_event = event_name
                        break

    groups.append(current_group)
    lengths.append(t0 + length - current_t0)
    divided_groups, current_group = [], []
    divided_lengths, current_length = [], 0
    divided_events, current_event = [], None
    for length, group, event_name in zip(lengths, groups, events):
        if (current_length + length) < analysis_period:
            current_group.extend(group)
            current_length += length
            if event_name is not None:
                current_event = event_name
        else:
            divided_groups.append(current_group)
            divided_lengths.append(current_length)
            divided_events.append(current_event)
            current_group, current_length, current_event = (
                [group],
                length,
                event_name,
            )

    divided_groups.append(current_group)
    divided_lengths.append(current_length)
    logging.info(
        "Analyzing {} segments of lengths {}".format(
            len(divided_lengths), divided_lengths
        )
    )

    for event_name, fnames, length in zip(
        divided_events, divided_groups, divided_lengths
    ):
        t0 = fname_re.search(str(fnames[0])).group("t0")
        if event_name is None:
            logging.info(
                "Building max {}s of background samples from timesliding "
                "{}s of data beginning at GPS time {}".format(
                    max_tb, length, t0
                )
            )
            analyzed_fnames, Tb, min_value, max_value = build_background(
                data_dir,
                write_dir,
                num_bins=num_bins,
                window_length=window_length,
                fnames=fnames,
                num_proc=8,
                norm_seconds=norm_seconds,
                max_tb=max_tb,
            )
        else:
            logging.info("Insert analysis here")


if __name__ == "__main__":
    main()
