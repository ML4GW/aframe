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
    group = luigi.Parameter(default=os.getenv("WANDB_RUN_GROUP", ""))
    tags = luigi.Parameter(default=os.getenv("WANDB_TAGS", ""))
