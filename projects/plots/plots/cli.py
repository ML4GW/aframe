import jsonargparse

from plots.main import sensitive_volume


def main(args=None):
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(sensitive_volume)
    cfg = parser.parse_args(args)
    sensitive_volume(**vars(cfg))


if __name__ == "__main__":
    main()
