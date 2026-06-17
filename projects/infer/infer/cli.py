import json
import os

import h5py
import jsonargparse
import numpy as np
from hermes.aeriel.client import InferenceClient
from utils.logging import configure_logging

from infer.data import Sequence
from infer.main import infer
from infer.postprocess import Postprocessor


def build_parser():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--return_timeseries", type=bool, default=False)

    parser.add_class_arguments(InferenceClient, "client")
    parser.add_class_arguments(Sequence, "data")
    parser.add_class_arguments(Postprocessor, "postprocessor")

    parser.link_arguments("data", "client.callback", apply_on="instantiate")
    parser.link_arguments(
        "data.inference_sampling_rate",
        "postprocessor.inference_sampling_rate",
        apply_on="parse",
    )
    parser.link_arguments(
        "data.t0", "postprocessor.t0", apply_on="instantiate"
    )
    parser.link_arguments(
        "data.shifts", "postprocessor.shifts", apply_on="parse"
    )

    return parser


def main(args=None):
    parser = build_parser()
    cfg = parser.parse_args(args)

    os.makedirs(cfg.outdir, exist_ok=True)
    if cfg.logfile is not None:
        os.makedirs(os.path.dirname(cfg.logfile) or ".", exist_ok=True)
    configure_logging(cfg.logfile, verbose=cfg.verbose)

    cfg = parser.instantiate_classes(cfg)
    with cfg.client:
        background, foreground, background_ts, foreground_ts = infer(
            cfg.client, cfg.data, cfg.postprocessor
        )

    background.write(os.path.join(cfg.outdir, "background.hdf5"))
    foreground.write(os.path.join(cfg.outdir, "foreground.hdf5"))
    with open(os.path.join(cfg.outdir, "metadata.json"), "w") as f:
        json.dump(
            {
                "background_length": len(background),
                "foreground_length": len(foreground),
            },
            f,
        )

    if cfg.return_timeseries:
        with h5py.File(os.path.join(cfg.outdir, "timeseries.hdf5"), "w") as f:
            # t0: segment start, for identifying the segment.
            # sample_t0: GPS time of the first timeseries sample,
            # offset by the postprocessor
            f.attrs["t0"] = cfg.data.t0
            f.attrs["sample_t0"] = cfg.postprocessor.t0
            f.attrs["inference_sampling_rate"] = (
                cfg.postprocessor.inference_sampling_rate
            )
            f.attrs["shifts"] = cfg.postprocessor.shifts
            f.create_dataset("background", data=background_ts)
            f.create_dataset(
                "foreground",
                data=foreground_ts
                if foreground_ts is not None
                else np.zeros(0),
            )


if __name__ == "__main__":
    main()
