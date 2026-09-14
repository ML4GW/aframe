from dataclasses import dataclass

# Character width of the ASCII window-placement diagram in __str__. The bar
# rows carry a 2-space indent and a trailing label (~27 chars), so the full
# line is ~width+27; 90 keeps it within a 120-column terminal.
DIAGRAM_WIDTH = 90


@dataclass
class WindowConfig:
    """
    Configuration for waveform window placement relative to the merger.

    The window's *leading edge* is its right edge — the edge closest to the
    merger in the scanning direction. Times are in seconds relative to the
    merger (t=0). Negative values mean the leading edge is before the merger
    (pre-merger / inspiral-only window); values larger than ``kernel_length``
    mean the window is entirely after the merger (post-merger window).

    Args:
        kernel_length:
            Length in seconds of the time-domain kernel passed to the model
            after whitening.
        window_lead_min:
            Earliest (smallest) time of the window leading edge relative to
            the merger. Equivalent to ``right_pad``. Must satisfy
            ``window_lead_min >= -fduration / 2`` at runtime so that
            ``right_pad_size`` remains non-negative.
        window_lead_max:
            Latest (largest) time of the window leading edge relative to the
            merger. Equivalent to ``kernel_length - left_pad``. Defaults to
            ``window_lead_min`` (fixed window position, no sliding). Must
            satisfy ``window_lead_max <= kernel_length + fduration / 2``
            at runtime so that ``left_pad_size`` remains non-negative.

    The valid sliding range of the window is
    ``window_lead_max - window_lead_min`` seconds.
    """

    kernel_length: float
    window_lead_min: float
    window_lead_max: float | None = None

    def __post_init__(self):
        if self.window_lead_max is None:
            self.window_lead_max = self.window_lead_min
        if self.window_lead_min > self.window_lead_max:
            raise ValueError(
                "Required: window_lead_min <= window_lead_max, got "
                f"window_lead_min={self.window_lead_min}, "
                f"window_lead_max={self.window_lead_max}"
            )

    @property
    def right_pad(self) -> float:
        """Minimum gap (seconds) between merger and window right edge."""
        return self.window_lead_min

    @property
    def left_pad(self) -> float:
        """Minimum gap (seconds) between window left edge and merger."""
        return self.kernel_length - self.window_lead_max

    @property
    def sliding_range(self) -> float:
        """Total range (seconds) over which the window can slide."""
        return self.window_lead_max - self.window_lead_min

    def __str__(self) -> str:
        width = DIAGRAM_WIDTH
        # Always include t=0 (merger) in the display range
        t0 = min(self.window_lead_min - self.kernel_length, 0.0)
        t1 = max(self.window_lead_max, 0.0)
        span = t1 - t0

        def pos(t):
            return max(0, min(width - 1, int((t - t0) / span * (width - 1))))

        merger = pos(0.0)

        def bar(start_t, end_t):
            s, e = pos(start_t), pos(end_t)
            row = [" "] * width
            row[s] = "["
            row[e] = "]"
            for i in range(s + 1, e):
                row[i] = "="
            return "".join(row)

        axis = ["-"] * width
        for t in (t0, 0.0, t1):
            axis[pos(t)] = "|"
        tick = " " * merger + "^"

        kl = self.kernel_length
        bar_min = bar(self.window_lead_min - kl, self.window_lead_min)
        bar_max = bar(self.window_lead_max - kl, self.window_lead_max)

        return "\n".join(
            [
                "Window placement relative to merger (t=0):",
                f"  kernel={kl:.2f}s  "
                f"sliding_range={self.sliding_range:.2f}s  "
                f"leading_edge=["
                f"{self.window_lead_min:+.3f}s, "
                f"{self.window_lead_max:+.3f}s]",
                "",
                f"  {''.join(axis)}",
                f"  {bar_min}  lead={self.window_lead_min:+.3f}s (earliest)",
                f"  {bar_max}  lead={self.window_lead_max:+.3f}s (latest)",
                f"  {tick} merger",
            ]
        )
