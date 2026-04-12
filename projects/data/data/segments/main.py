import logging

from data.segments.segments import DataQualityDict


def main(args):
    args_dict = {k: v for k, v in args.as_dict().items() if k != "config"}
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
