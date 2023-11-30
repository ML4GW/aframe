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
    cpus = luigi.IntParameter(default=8)
    gpus = luigi.IntParameter(default=1)
    memory = luigi.Parameter(default="10G")
    min_gpu_memory = luigi.IntParameter(default=0)


class ray_head(luigi.Config):
    cpus = luigi.IntParameter(default=2)
    memory = luigi.Parameter(default="1G")

class data(luigi.Config):
    kernel_length = luigi.FloatParameter(default=1.5)
    inference_sampling_rate = luigi.FloatParameter(default=16)
    sample_rate = luigi.FloatParameter(default=2048)
    inference_batch_size = luigi.IntParameter(default=512)
    fduration = luigi.FloatParameter(default=2)
    inference_psd_length = luigi.FloatParameter(default=64)
    fftlength = luigi.FloatParameter(default=2)
    highpass = luigi.FloatParameter(default=32)

class export(luigi.Config):
    streams_per_gpu = luigi.IntParameter(default=1)
    aframe_instances = luigi.IntParameter(default=None)
    platform = luigi.Parameter(default="onnx")
    clean = luigi.BoolParameter(default=False)


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
    data = data()
    export = export()

class Defaults:
    TRAIN = os.path.join(project_base, "train", "config.yaml")
    EXPORT = os.path.join(project_base, "export", "config.yaml")
