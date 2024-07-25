import os

import luigi
from luigi.contrib.s3 import S3Client

from aframe.parameters import PathParameter

project_base = "/opt/aframe/projects"


class ray_worker(luigi.Config):
    replicas = luigi.IntParameter(
        default=1, description="Number of ray worker replicas to deploy"
    )
    gpus_per_replica = luigi.IntParameter(
        default=2,
        description="Number of gpus to allocate to each ray worker replica",
    )
    cpus_per_gpu = luigi.IntParameter(
        default=12,
        description="Number of cpus to allocate to the ray worker per gpu",
    )
    memory_per_gpu = luigi.FloatParameter(
        default=70,
        description="Amount of (CPU) memory to allocate per GPU in GB"
        "to each ray worker deployment. For tuning jobs currently the "
        "data is downloaded for each trial.",
    )
    min_gpu_memory = luigi.Parameter(
        default="15000",
        description="Minimum amount of memory each gpu should have",
    )

    @property
    def memory_per_replica(self):
        return f"{self.memory_per_gpu * self.gpus_per_replica}G"

    @property
    def cpus_per_replica(self):
        return self.cpus_per_gpu * self.gpus_per_replica


class ray_head(luigi.Config):
    cpus = luigi.IntParameter(
        default=32,
        description="Number of cpus to allocate to the ray head deployment",
    )
    memory = luigi.Parameter(
        default="32G",
        description="Amount of memory in GB "
        "to allocate to the ray head deployment",
    )


class wandb(luigi.Config):
    api_key = luigi.Parameter(default=os.getenv("WANDB_API_KEY", ""))
    entity = luigi.Parameter(default=os.getenv("WANDB_ENTITY", ""))
    project = luigi.Parameter(default=os.getenv("WANDB_PROJECT", "aframe"))
    name = luigi.Parameter(default=os.getenv("WANDB_NAME", ""))
    group = luigi.Parameter(default=os.getenv("WANDB_RUN_GROUP", ""))
    tags = luigi.Parameter(default=os.getenv("WANDB_TAGS", ""))
    username = luigi.Parameter(default=os.getenv("WANDB_USERNAME", ""))


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
        return S3Client(endpoint_url=self.endpoint_url)


class Defaults:
    TRAIN = os.path.join(project_base, "train", "config.yaml")


# mapping from external to internal nautilus urls
nautilus_urls = {
    # after chatting with computing folks from nautilus,
    # learned that internal url was experiencing transient permission issues,
    # so just map from external to external for the time being
    # (which apparently is slower, but will still work) before this is fixed
    "https://s3-west.nrp-nautilus.io": "https://s3-west.nrp-nautilus.io",
    "https://s3-central.nrp-nautilus.io": "http://rook-ceph-rgw-centrals3.rook-central",  # noqa: E501
    "https://s3-east.nrp-nautilus.io": "http://rook-ceph-rgw-easts3.rook-east",
}


class paths(luigi.Config):
    train_datadir = PathParameter(default=os.getenv("AFRAME_TRAIN_DATA_DIR"))
    train_rundir = PathParameter(default=os.getenv("AFRAME_TRAIN_RUN_DIR"))
    results_dir = PathParameter(default=os.getenv("AFRAME_RESULTS_DIR"))
    test_datadir = PathParameter(default=os.getenv("AFRAME_TEST_DATA_DIR"))
    condor_dir = PathParameter(default=os.getenv("AFRAME_CONDOR_DIR"))
    tmp_dir = PathParameter(default=os.getenv("AFRAME_TMPDIR"))
