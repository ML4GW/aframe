import os

import luigi


class wandb(luigi.Config):
    api_key = luigi.Parameter(default=os.getenv("WANDB_API_KEY", ""))
    entity = luigi.Parameter(default=os.getenv("WANDB_ENTITY", ""))
    project = luigi.Parameter(default=os.getenv("WANDB_PROJECT", "aframe"))
    name = luigi.Parameter(default=os.getenv("WANDB_NAME", ""))
    group = luigi.Parameter(default=os.getenv("WANDB_GROUP", ""))
    tags = luigi.Parameter(default=os.getenv("WANDB_TAGS", ""))
