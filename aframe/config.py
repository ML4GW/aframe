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
