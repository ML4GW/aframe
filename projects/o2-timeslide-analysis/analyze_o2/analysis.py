from typing import Optional

from hermes.typeo import typeo

from analyze_o2 import analysis
from bbhnet.analysis import distributions, matched_filter


@typeo
def main(
    data_dir: str,
    write_dir: str,
    num_bins: int = 10000,
    window_length: float = 1.0,
    norm_seconds: Optional[float] = None,
    max_tb: Optional[float] = None
):
    t0s, lengths = [], []
    for t0, length in zip(t0s, lengths):
        fnames, min_value, max_value = analysis.build_background(
            data_dir,
            write_dir,
            num_bins=num_bins,
            window_length=window_length,
            num_proc=8,
            t0=t0,
            length=length,
            norm_seconds=norm_seconds,
            max_tb=max_tb
        )


if __name__ == "__main__":
    main()
