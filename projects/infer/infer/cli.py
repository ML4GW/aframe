import os

import jsonargparse
from infer.data import Sequence
from infer.main import infer
from infer.postprocess import Postprocessor

from aframe.logging import configure_logging
from hermes.aeriel.client import InferenceClient


def build_parser():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)

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

    if cfg.outdir is not None:
        os.makedirs(cfg.outdir, parents=True, exist_ok=True)

    if cfg.logfile is not None:
        logdir = os.path.dirname(cfg.logfile)
        os.makedirs(logdir, parents=True, exist_ok=True)
    configure_logging(cfg.logfile, verbose=cfg.verbose)

    cfg = parser.instantiate_classes(cfg)
    with cfg.client:
        background, foreground = infer(cfg.client, cfg.data, cfg.postprocessor)

    if cfg.outdir is not None:
        background.write(os.path.join(cfg.outdir, "background.hdf5"))
        foreground.write(os.path.join(cfg.outdir, "foreground.hdf5"))
    else:
        return background, foreground


if __name__ == "__main__":
    main()
