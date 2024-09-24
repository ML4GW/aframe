import logging

from jsonargparse import ArgumentParser

from data.segments.segments import DataQualityDict

parser = ArgumentParser()
parser.add_method_arguments(DataQualityDict, "query_segments")
parser.add_argument("--output_file", "-o", type=str)


def main(args=None):
    args = args.query.as_dict()
    output_file = args.pop("output_file")

    logging.info(
        "Querying active segments in interval ({start}, {end})".format(**args)
    )
    segments = DataQualityDict.query_segments(**args)

    logging.info(
        "Discovered {} valid segments, writing to {}".format(
            len(segments), output_file
        )
    )
    segments.write(output_file)
