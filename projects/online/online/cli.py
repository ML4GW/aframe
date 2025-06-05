import jsonargparse
from online.main import main


def build_parser():
    # use omegaconf to suppor env var interpolation
    parser = jsonargparse.ArgumentParser(parser_mode="omegaconf")
    parser.add_function_arguments(main)
    parser.add_argument("--config", action="config")

    parser.link_arguments(
        "inference_params",
        "amplfi_hl_architecture.init_args.num_params",
        compute_fn=lambda x: len(x),
        apply_on="parse",
    )

    parser.link_arguments(
        "inference_params",
        "amplfi_hlv_architecture.init_args.num_params",
        compute_fn=lambda x: len(x),
        apply_on="parse",
    )

    return parser


def cli(args=None):
    parser = build_parser()
    args = parser.parse_args(args)
    args.pop("config")
    args = parser.instantiate_classes(args)
    main(**vars(args))


if __name__ == "__main__":
    cli()
