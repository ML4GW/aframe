import os

import luigi

project_base = "/opt/aframe/projects"

# base config that stores parameters 
# common to multiple tasks
class base(luigi.Config):
    ifos = luigi.ListParameter(default=["H1", "L1"])
    run_dir = luigi.Parameter(default=os.getenv("RUN_DIR", ""))
    data_dir = luigi.Parameter(default=os.getenv("DATA_DIR", ""))
    kernel_length = luigi.FloatParameter(default=1.5)
    inference_sampling_rate = luigi.FloatParameter(default=16)
    sample_rate = luigi.FloatParameter(default=2048)
    batch_size = luigi.IntParameter(default=512)
    inference_batch_size = luigi.IntParameter(default=512)
    fduration = luigi.FloatParameter(default=2)
    inference_psd_length = luigi.FloatParameter(default=64)
    fftlength = luigi.FloatParameter(default=2.0)
    highpass = luigi.FloatParameter(default=32.0)
    
    @property
    def logdir(self):
        return os.path.join(self.run_dir , "logs")

    @property
    def num_ifos(self):
        return len(self.ifos)


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

# config for export task
# some parameters inherit a default from base. 
# Still make them luigi.Parameters to accommodate use cases 
# where one might want to override the base default.
 
class export(luigi.Config):
    fftlength = luigi.FloatParameter(default=base().fftlength)
    fduration = luigi.FloatParameter(default=base().fduration)
    kernel_length = luigi.FloatParameter(default=base().kernel_length)
    inference_sampling_rate = luigi.FloatParameter(default=base().inference_sampling_rate)
    sample_rate = luigi.FloatParameter(default=base().sample_rate)
    fduration = luigi.FloatParameter(default=base().fduration)
    repository_directory = luigi.Parameter(default=os.path.join(base().run_dir, "model_repo"))
    streams_per_gpu = luigi.IntParameter(default=2)
    aframe_instances = luigi.IntParameter(default=2)
    # TODO: resolve enum platform parsing error
    #platform = luigi.Parameter(default="TENSORRT")
    clean = luigi.BoolParameter(default=False)
    num_ifos = luigi.IntParameter(default=base().num_ifos)
    batch_size = luigi.IntParameter(default=base().inference_batch_size)
    psd_length = luigi.FloatParameter(default=base().inference_psd_length)
    highpass = luigi.FloatParameter(default=base().highpass)
    logfile = luigi.Parameter(default=os.path.join(base().logdir, "export.log"))


class train(luigi.Config):
    ifos = luigi.ListParameter(default=base().ifos)

class aframe(luigi.Config):
    """
    Global config for aframe experiments
    """
    container_root = luigi.Parameter(
        default=os.getenv(
            "AFRAME_CONTAINER_ROOT", os.path.expanduser("~/aframe/images")
        )
    )
    wandb = wandb()
    s3 = s3()
    ray_worker = ray_worker()
    ray_head = ray_head()
    export = export()
    train = train()


class Defaults:
    TRAIN = os.path.join(project_base, "train", "config.yaml")
    EXPORT = os.path.join(project_base, "export", "config.yaml")
