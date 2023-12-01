import os

import luigi

project_base = "/opt/aframe/projects"


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


class ray_worker(luigi.Config):
    replicas = luigi.IntParameter(default=2)
    gpus = luigi.IntParameter(default=1)
    cpus_per_gpu = luigi.IntParameter(default=8)
    memory = luigi.Parameter(default="10G")
    min_gpu_memory = luigi.IntParameter(default=0)


class ray_head(luigi.Config):
    cpus = luigi.IntParameter(default=2)
    memory = luigi.Parameter(default="1G")


class aframe(luigi.Config):
    """
    Global config for aframe experiments
    """

    ifos = luigi.ListParameter(default=["H1", "L1"])
    container_root = luigi.Parameter(
        default=os.getenv(
            "AFRAME_CONTAINER_ROOT", os.path.expanduser("~/aframe/images")
        )
    )

    wandb = wandb()
    s3 = s3()
    ray_worker = ray_worker()
    ray_head = ray_head()


class Defaults:
    TRAIN = os.path.join(project_base, "train", "config.yaml")
