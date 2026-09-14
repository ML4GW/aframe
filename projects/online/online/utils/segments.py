import json
from enum import StrEnum
from pathlib import Path

from online.utils.timing import gps_now

CURRENT_FILE = "current.json"
SEGMENTS_FILE = "segments.txt"

COLUMNS = ["state", "start", "stop", "ifos_ready"]
HEADER = ",".join(COLUMNS) + "\n"


class PipelineState(StrEnum):
    ANALYZING = "analyzing"
    WARMUP = "warmup"
    STARTUP = "startup"
    NOT_READY = "not_ready"
    MISSING_DATA = "missing_data"
    SEARCH_DOWN = "search_down"


def pipeline_state(hl_ready: bool, full_psd_present: bool) -> str:
    """
    Which state a block of data that reached the search was handled
    in: either we searched it, or the data wasn't analysis ready, or
    it was but our own whitening filter wasn't ready for it yet.
    """
    if not hl_ready:
        return PipelineState.NOT_READY
    return (
        PipelineState.ANALYZING if full_psd_present else PipelineState.WARMUP
    )


class SegmentWriter:
    """
    Record of which state the search was in while running.

    Cosecutive blocks in the same state are coalesced into a single
    segment, which is written out only once the state changes or the
    data stream becomes discontiguous. The currently open segment is
    also sent to a heartbeat file so that a monitoring process can tell
    whether the search is actively running.

    All times are in GPS seconds.

    Args:
        outdir:
            Directory that the search writes its output to. Segments
            are written to a `segments` subdirectory of this.
        block_duration:
            Length in seconds of the blocks handed to `update`.
        process_start:
            GPS time when the search process started, or None to use
            the current time
        heartbeat_cadence:
            Minimum number of seconds between updates of the heartbeat
            file.
    """

    def __init__(
        self,
        outdir: Path,
        block_duration: float,
        process_start: float | None = None,
        heartbeat_cadence: float = 1.0,
    ) -> None:
        self.directory = outdir / "segments"
        self.directory.mkdir(parents=True, exist_ok=True)

        self.current_file = self.directory / CURRENT_FILE
        self.segments_file = self.directory / SEGMENTS_FILE

        self.block_duration = block_duration
        self.heartbeat_cadence = heartbeat_cadence

        if not self.segments_file.exists():
            with open(self.segments_file, "w") as f:
                f.write(HEADER)

        # Clear out any leftover heartbeat file
        if self.current_file.exists():
            self._flush_current_file()

        self.state = None
        self.start = None
        self.stop = None
        self.ifos_ready = ""
        self.last_heartbeat = 0.0

        self.process_start = (
            gps_now() if process_start is None else process_start
        )
        self._append(
            PipelineState.STARTUP,
            self.process_start,
            self.process_start,
            "",
        )

    def _flush_current_file(self) -> None:
        with open(self.current_file, "r") as f:
            current_data = json.load(f)
        self._append(
            current_data["state"],
            current_data["start"],
            current_data["stop"],
            current_data["ifos_ready"],
        )
        self.current_file.unlink()

    def _append(
        self,
        state: PipelineState,
        start: float,
        stop: float,
        ifos_ready: str,
    ) -> None:
        with open(self.segments_file, "a") as f:
            f.write(f"{state},{start:.5f},{stop:.5f},{ifos_ready}\n")

    def _write_heartbeat(self, now: float) -> None:
        with open(self.current_file, "w") as f:
            json.dump(
                {
                    "state": self.state,
                    "start": self.start,
                    "stop": self.stop,
                    "ifos_ready": self.ifos_ready,
                    "heartbeat": now,
                },
                f,
            )
        self.last_heartbeat = now

    def update(
        self,
        state: PipelineState,
        t0: float,
        ready: list[bool] = None,
    ) -> None:
        """
        Update the current segment with a new block of data.

        Args:
            state: The pipeline state for the new block.
            t0: The GPS start time of the new block.
            ready:
                A list of booleans indicating which interferometers
                were ready.
        """
        ifos_ready = (
            "".join("1" if r else "0" for r in ready)
            if ready is not None
            else ""
        )

        now = gps_now()
        contiguous = self.stop is not None and abs(t0 - self.stop) < 1e-6
        same_segment = (self.state == state) and ifos_ready == self.ifos_ready
        if contiguous and same_segment:
            self.stop = t0 + self.block_duration
            if now - self.last_heartbeat > self.heartbeat_cadence:
                self._write_heartbeat(now)
        else:
            if self.state is not None:
                self._append(
                    self.state,
                    self.start,
                    self.stop,
                    self.ifos_ready,
                )
            self.state = state
            self.start = t0
            self.stop = t0 + self.block_duration
            self.ifos_ready = ifos_ready
            self._write_heartbeat(now)

    def close(self) -> None:
        """
        Close the segment and flush any remaining data.
        """
        if self.state is not None:
            self._append(
                self.state,
                self.start,
                self.stop,
                self.ifos_ready,
            )
        if self.current_file.exists():
            self.current_file.unlink()
