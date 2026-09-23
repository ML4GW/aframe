import logging
from pathlib import Path

from data.fetch.fetch import fetch


def main(args):
    args_dict = {k: v for k, v in args.as_dict().items() if k != "config"}
    output_file = Path(args_dict.pop("output_file"))

    X = fetch(**args_dict)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing downloaded data to {output_file}")
    X.write(output_file, format="hdf5")
