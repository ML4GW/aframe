import os

import luigi

project_base = "/opt/aframe/projects"


class ray_worker(luigi.Config):
    replicas = luigi.IntParameter(default=2)
    gpus = luigi.IntParameter(default=1)
    cpus_per_gpu = luigi.IntParameter(default=8)
    memory = luigi.Parameter(default="10G")
    min_gpu_memory = luigi.IntParameter(default=0)


class ray_head(luigi.Config):
    cpus = luigi.IntParameter(default=2)
    memory = luigi.Parameter(default="1G")


class Defaults:
    TRAIN = os.path.join(project_base, "train", "config.yaml")
    EXPORT = os.path.join(project_base, "export", "config.yaml")


class s3(luigi.Config):
    endpoint_url = luigi.Parameter(default=os.getenv("AWS_ENDPOINT_URL"))
    credentials = luigi.Parameter(
        default=os.path.expanduser("~/.aws/credentials")
    )


# mapping from external to internal nautilus urls
nautilus_urls = {
    "https://s3-west.nrp-nautilus.io": "http://rook-ceph-rgw-nautiluss3.rook",
    "https://s3-central.nrp-nautilus.io": "http://rook-ceph-rgw-centrals3.rook-central",  # noqa: E501
    "https://s3-east.nrp-nautilus.io": "http://rook-ceph-rgw-easts3.rook-east",
}
