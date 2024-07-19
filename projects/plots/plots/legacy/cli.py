import jsonargparse
from plots.legacy.main import main as calc_sensitive_volume


def main(args=None):
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(calc_sensitive_volume)
    args = parser.parse_args()
    calc_sensitive_volume(**vars(args))


if __name__ == "__main__":
    main()
