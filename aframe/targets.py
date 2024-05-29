from typing import Literal

import luigi
from luigi.contrib.s3 import S3Target
from luigi.format import BaseWrapper, WrappedFormat

from aframe.config import s3
from aframe.parameters import PATH_LIKE


# format for writing h5 files to s3
# via bytes streams
class BytesFormat(WrappedFormat):
    input = "bytes"
    output = "bytes"
    wrapper_cls = BaseWrapper


Bytes = BytesFormat()


# law targets are currently not
# passing format correctly so do this for now
class LawS3Target(S3Target):
    optional = False

    def complete(self):
        return self.exists()


class LawLocalTarget(luigi.LocalTarget):
    optional = False

    def complete(self):
        return self.exists()


# use s3 if path starts with s3://,
# otherwise use a local target
def s3_or_local(path: PATH_LIKE, format: Literal["hdf5", "txt"] = "hdf5"):
    format = Bytes if format == "hdf5" else None
    path = str(path)
    if path.startswith("s3://"):
        return LawS3Target(
            path,
            client=s3().client,
            ContentType="application/octet-stream",
            format=format,
        )
    else:
        return LawLocalTarget(path, format=format)
