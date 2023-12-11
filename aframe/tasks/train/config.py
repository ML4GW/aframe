import os

import luigi


class wandb(luigi.Config):
    api_key = luigi.Parameter(default=os.getenv("WANDB_API_KEY", ""))
    entity = luigi.Parameter(default=os.getenv("WANDB_ENTITY", ""))
    project = luigi.Parameter(default=os.getenv("WANDB_PROJECT", "aframe"))
    name = luigi.Parameter(default=os.getenv("WANDB_NAME", ""))
    group = luigi.Parameter(default=os.getenv("WANDB_GROUP", ""))
    tags = luigi.Parameter(default=os.getenv("WANDB_TAGS", ""))


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
