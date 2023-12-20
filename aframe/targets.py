import luigi
from luigi.contrib.s3 import S3Target
from luigi.format import BaseWrapper, WrappedFormat


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
def s3_or_local(path, client):
    if path.startswith("s3://"):
        return LawS3Target(
            path,
            client=client,
            ContentType="application/octet-stream",
            format=Bytes,
        )
    else:
        return LawLocalTarget(path, format=Bytes)
