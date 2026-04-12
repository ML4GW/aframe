import jsonargparse

from plots.legacy.main import main as calc_sensitive_volume


def main(args=None):
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(calc_sensitive_volume)
    cfg = parser.parse_args(args)
    calc_sensitive_volume(**vars(cfg))


if __name__ == "__main__":
    main()
