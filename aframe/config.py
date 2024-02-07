import os

import luigi
from luigi.contrib.s3 import S3Client

project_base = "/opt/aframe/projects"


class ray_worker(luigi.Config):
    replicas = luigi.IntParameter(default=2)
    gpus_per_replica = luigi.IntParameter(default=2)
    cpus_per_gpu = luigi.IntParameter(default=8)
    gpus_per_worker = luigi.IntParameter(default=1)
    memory = luigi.Parameter(default="10G")
    min_gpu_memory = luigi.Parameter(default="15000")


class ray_head(luigi.Config):
    cpus = luigi.IntParameter(default=2)
    memory = luigi.Parameter(default="1G")


class s3(luigi.Config):
    endpoint_url = luigi.Parameter(default=os.getenv("AWS_ENDPOINT_URL"))
    aws_access_key_id = luigi.Parameter(default=os.getenv("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key = luigi.Parameter(
        default=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    def get_s3_credentials(self):
        keys = ["aws_access_key_id", "aws_secret_access_key"]
        secret = {}
        for key in keys:
            secret[key.upper()] = getattr(self, key)
        return secret

    def get_internal_s3_url(self):
        # if user specified an external nautilus url,
        # map to the corresponding internal url,
        # since the internal url is what is used by the
        # kubernetes cluster to access s3
        url = self.endpoint_url
        if url in nautilus_urls:
            return nautilus_urls[url]
        return url

    @property
    def client(self):
        S3Client(endpoint_url=self.endpoint_url)


class Defaults:
    TRAIN = os.path.join(project_base, "train", "config.yaml")


# mapping from external to internal nautilus urls
nautilus_urls = {
    "https://s3-west.nrp-nautilus.io": "http://rook-ceph-rgw-nautiluss3.rook",
    "https://s3-central.nrp-nautilus.io": "http://rook-ceph-rgw-centrals3.rook-central",  # noqa: E501
    "https://s3-east.nrp-nautilus.io": "http://rook-ceph-rgw-easts3.rook-east",
}
