import logging

from data.segments.segments import DataQualityDict
from jsonargparse import ActionConfigFile, ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", action=ActionConfigFile)
parser.add_method_arguments(DataQualityDict, "query_segments")
parser.add_argument("--output_file", "-o", type=str)


def main(args=None):
    cfg = parser.parse_args(args)
    args_dict = {k: v for k, v in cfg.as_dict().items() if k != "config"}

    output_file = args_dict.pop("output_file")

    logging.info(
        "Querying active segments in interval ({start}, {end})".format(
            **args_dict
        )
    )
    segments = DataQualityDict.query_segments(**args_dict)

    logging.info(
        f"Discovered {len(segments)} valid segments, writing to {output_file}"
    )
    segments.write(output_file)
