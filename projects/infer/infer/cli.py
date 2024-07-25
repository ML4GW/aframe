import os

import jsonargparse
from infer.data import Sequence
from infer.main import infer
from infer.postprocess import Postprocessor

from hermes.aeriel.client import InferenceClient
from utils.logging import configure_logging


def build_parser():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--force", type=bool, default=False)

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
        os.makedirs(cfg.outdir, exist_ok=True)

    if cfg.logfile is not None:
        logdir = os.path.dirname(cfg.logfile)
        os.makedirs(logdir, exist_ok=True)
    configure_logging(cfg.logfile, verbose=cfg.verbose)

    background_path = os.path.join(cfg.outdir, "background.hdf5")
    foreground_path = os.path.join(cfg.outdir, "foreground.hdf5")

    cfg = parser.instantiate_classes(cfg)
    with cfg.client:
        background, foreground = infer(cfg.client, cfg.data, cfg.postprocessor)

    if cfg.outdir is not None:
        background.write(background_path)
        foreground.write(foreground_path)
    else:
        return background, foreground


if __name__ == "__main__":
    main()
