import os

import luigi

# TODO:
# default compute requirements calibrated based on
# test runs on nautilus
CPUS_PER_GPU = 8


class wandb(luigi.Config):
    api_key = luigi.Parameter(default=os.getenv("WANDB_API_KEY", ""))
    entity = luigi.Parameter(default=os.getenv("WANDB_ENTITY", ""))
    project = luigi.Parameter(default=os.getenv("WANDB_PROJECT", "aframe"))
    name = luigi.Parameter(default=os.getenv("WANDB_NAME", ""))
    group = luigi.Parameter(default=os.getenv("WANDB_GROUP", ""))
    tags = luigi.Parameter(default=os.getenv("WANDB_TAGS", ""))


class train_remote(luigi.Config):
    image = luigi.Parameter(default="ghcr.io/ml4gw/aframev2/train:main")
    min_gpu_memory = luigi.IntParameter(default=15000)
    request_gpus = luigi.IntParameter(default=8)
    request_cpus = luigi.IntParameter(default=60)
    request_cpu_memory = luigi.Parameter(default="128Gi")
