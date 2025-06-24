import os
import numpy as np
from pathlib import Path
from online.dataloading.utils import (
    build_resample_filter,
    resample,
    parse_frame_name,
    GWF_SAMPLE_RATE,
)
from typing import List, Optional, Generator
from gwpy.timeseries import TimeSeriesDict
import logging
import torch
import multiprocessing as mp


STATE_VECTOR_SAMPLE_RATE = 16


class OfflineFrameFileLoader:
    def __init__(
        self,
        datadir: Path,
        ifo_suffix: str,
        ifos: list[str],
        strain_channels: list[str],
        state_channels: list[str],
    ):
        """
        Initialize the chunk generator for timeseries data.

        Args:
            directories: List of directory paths containing timeseries files
            sample_rate: Sample rate of the data in Hz
        """
        self.strain_channels = strain_channels
        self.state_channels = state_channels
        self.directories: dict[str, Path] = {
            ifo: datadir / f"{ifo}_{ifo_suffix}" for ifo in ifos
        }
        self.ifos = ifos
        self.frame_metadata = {}
        self.start, self.end = self.parse_directories()

    def parse_directories(self):
        """Scan directories and organize file information by timestamp."""
        for ifo, path in self.directories.items():
            self.frame_metadata[ifo] = {}

            for filename in path.iterdir():
                if filename.suffix == ".gwf":
                    _, timestamp, duration = parse_frame_name(filename)
                    self.frame_metadata[ifo][timestamp] = {
                        "path": os.path.join(path, filename),
                        "end": timestamp + duration,
                    }

        # Find the common time range across all directories
        return self.find_intersection()

    def find_intersection(self):
        """Find the overlapping time range across all ifos"""
        ranges = []

        for ifo, frame_metadata in self.frame_metadata.items():
            starts = list(frame_metadata.keys())
            start = min(starts)
            last_timestamp = max(starts)
            end = frame_metadata[last_timestamp]["end"]
            logging.info(
                f"Data in {self.directories[ifo]} spans {start} to {end}"
            )
            ranges.append((start, end))

        start = max(r[0] for r in ranges)
        end = min(r[1] for r in ranges)

        logging.info(f"Analyzing intersection: {start} to {end}")
        return start, end

    def find_frames(self, ifo: str, start: float, end: float) -> List[str]:
        """
        Find all files that overlap with the given time range.
        """
        frames = []
        for path in self.directories[ifo].iterdir():
            _, f_start, duration = parse_frame_name(path)
            f_end = duration + f_start
            if f_start < end and f_end > start:
                frames.append(path)

        # sort by timestamp (extracted from filename)
        frames.sort(key=lambda path: parse_frame_name(path)[1])
        return frames

    def read_data(
        self,
        ifo: str,
        strain_channel: str,
        state_channel,
        start: float,
        end: float,
    ) -> TimeSeriesDict:
        """
        Load data for a specified time range
        potentially spanning multiple files.
        """
        # find frames for this ifo that contain the span [start, end]
        frames = self.find_frames(ifo, start, end)

        # read in both strain and state vector channels
        data = TimeSeriesDict.read(
            frames,
            channels=[strain_channel, state_channel],
            start=start,
            end=end,
        )

        state = data.pop(state_channel)
        sample_rate = state.sample_rate.value
        state = state.value

        # TODO:
        # this logic here is hardcoded to Virgo
        # which has state vector sample rate of 1.
        if sample_rate != STATE_VECTOR_SAMPLE_RATE:
            state = np.repeat(state, STATE_VECTOR_SAMPLE_RATE)

        return data.pop(strain_channel), state

    def generate_chunks(
        self, chunk_length: int
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate chunks of data chronologically.
        """
        chunk_start = self.start

        while chunk_start + chunk_length <= self.end:
            chunk_end = min(self.end, chunk_start + chunk_length)

            logging.debug(f"Loading chunk {chunk_start} to {chunk_end}")
            strain = []
            state = []

            for ifo, strain_channel, state_channel in zip(
                self.ifos, self.strain_channels, self.state_channels
            ):
                ifo_strain, ifo_state = self.read_data(
                    ifo, strain_channel, state_channel, chunk_start, chunk_end
                )
                strain.append(ifo_strain.value)
                state.append(ifo_state)

            strain = np.stack(strain)
            state = np.stack(state)
            yield strain, state, chunk_start

            chunk_start = chunk_end


def chunk_loader_worker(
    datadir: Path,
    ifo_suffix: str,
    ifos: list[str],
    strain_channels: list[str],
    state_channels: list[str],
    chunk_size: int,
    queue: mp.Queue,
):
    """
    Worker function that runs in a separate process to load data chunks.
    """
    chunk_loader = OfflineFrameFileLoader(
        datadir, ifo_suffix, ifos, strain_channels, state_channels
    )

    for strain, state, t0 in chunk_loader.generate_chunks(chunk_size):
        queue.put((strain, state, t0))

    queue.put(None)


def offline_data_iterator(
    datadir: Path,
    channels: List[str],
    ifos: List[str],
    sample_rate: float,
    ifo_suffix: str = None,
    state_channels: Optional[dict[str, str]] = None,
) -> Generator[tuple[torch.Tensor, float, list[bool]], None, None]:
    """
    Similar to `data_iterator` above, but does not
    assume frame files arrive in 1 second frames.
    It does assume the frame files between different
    interferometers match in length.
    """

    # build resampling filter
    factor = GWF_SAMPLE_RATE / sample_rate
    if not factor.is_integer():
        raise ValueError(
            f"Specified sample rate {sample_rate} must "
            f"evenly divide the frame sample rate {GWF_SAMPLE_RATE}"
        )
    factor = int(factor)
    b, a = build_resample_filter(factor)

    frame_buffer = np.zeros((len(ifos), 0))
    # slice corresponds to middle second of
    # a 3 second buffer; the middle second is
    # yielded at each step to mitigate resampling
    # edge effects
    buffer_slc = slice(-int(2 * sample_rate), -int(sample_rate))

    last_ready = [True] * len(ifos)

    queue = mp.Queue()

    CHUNK_LENGTH = 4096
    # load chunks in separate process
    loader_process = mp.Process(
        target=chunk_loader_worker,
        args=(
            datadir,
            ifo_suffix,
            ifos,
            channels,
            list(state_channels.values()) if state_channels else [],
            CHUNK_LENGTH,
            queue,
        ),
    )

    loader_process.start()

    while True:
        chunk_data = queue.get()
        if chunk_data is None:
            break

        strain, state, t0 = chunk_data
        logging.info(f"Analysing chunk starting at {t0}")
        # now, mimic online deployment by yielding 1 second
        # increments from the pre-loaded data

        # yield one second per duration
        for i in range(int(CHUNK_LENGTH)):
            ready = [True] * len(ifos)
            logging.debug(f"Reading frames from timestamp {t0}")
            frames = []
            frame_slc = slice(
                int(i * GWF_SAMPLE_RATE), int((i + 1) * GWF_SAMPLE_RATE)
            )
            state_vector_slc = slice(int(i * 16), int((i + 1) * 16))
            for j, ifo in enumerate(ifos):
                frames.append(strain[j, frame_slc])
                # if state channels were specified,
                # check that the 3rd bit is on.
                # If left as `None` set ifo_ready
                # to True by default.
                # TODO: parameterize bitmask
                ifo_ready = True
                if state_channels is not None:
                    state_vector = state[j, state_vector_slc]
                    ifo_ready = ((state_vector & 3) == 3).all()

                # some useful logging
                # for when ifos enter and exit
                # analyis ready mode
                if not ifo_ready:
                    if last_ready[j]:
                        logging.info(f"IFO {ifo} exiting analysis ready mode")
                    else:
                        logging.debug(f"IFO {ifo} not analysis ready")
                else:
                    if not last_ready[j]:
                        logging.info(f"IFO {ifo} entering analysis ready mode")

                # mark this ifos readiness in array
                ready[j] &= ifo_ready

            else:
                logging.debug("Read successful")

                frame = np.stack(frames)
                frame_buffer = np.append(frame_buffer, frame, axis=1)
                dur = frame_buffer.shape[-1] / GWF_SAMPLE_RATE
                # Need at least 3 seconds to be able to crop out edge effects
                # from resampling and just yield the middle second
                if dur >= 3:
                    x = resample(frame_buffer, factor, b, a)
                    x = x[:, buffer_slc]
                    frame_buffer = frame_buffer[:, GWF_SAMPLE_RATE:]
                    # yield last_ready, which corresponds to
                    # the data quality bits of the previous second
                    # of data, i.e. the middle second of the
                    # buffer that is being yielded as well
                    yield torch.Tensor(x.copy()).double(), t0 - 1, last_ready

                last_ready = ready
                t0 += 1
