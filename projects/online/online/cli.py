import jsonargparse

from online.main import main


def build_parser():
    # use omegaconf to suppor env var interpolation
    parser = jsonargparse.ArgumentParser(parser_mode="omegaconf")
    parser.add_function_arguments(main, fail_untyped=False, sub_configs=True)
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

    # TODO: This is a workaround for linking sample_rate and
    # kernel_length argument between the augmentor and
    # the global parameters in the config.
    try:
        parser.link_arguments(
            "sample_rate",
            "augmentor.init_args.sample_rate",
            apply_on="parse",
        )
    except Exception:
        pass

    try:
        parser.link_arguments(
            "kernel_length",
            "augmentor.init_args.kernel_length",
            apply_on="parse",
        )
    except Exception:
        pass

    return parser


def cli(args=None):
    parser = build_parser()
    args = parser.parse_args(args)
    args.pop("config")
    args = parser.instantiate(args)
    main(**vars(args))


if __name__ == "__main__":
    cli()
