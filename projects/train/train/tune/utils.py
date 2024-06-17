"""
Hyperparameter tuning utilities based largely on this tutorial
https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html

since the APIs used in this tutorial
https://docs.ray.io/en/latest/tune/examples/tune-vanilla-pytorch-lightning.html

are out of date with the latest version of lightning.
(They import `from pytorch_lightning`, which doesn't
play well with the `from lightning import pytorch`, the
modern syntax that we use. This is insane, of course, but
that's just the way it is.)

The downside is that I can't figure out how to get local
tune jobs to use the correct `gpus_per_worker`. In my local
tests, one job just claims all the available GPUs no matter
what. This doesn't seem to be a problem for remote, which
is the use case we're largely targeting anyway, so I'm not
freaking out about it. But it would be nice to figure this out.
Unfortunately it does not seem as if this is a heavily trafficked
API, at least not the latest version, and so I've had some
difficulty finding resources from other people dealing with
this issue.
"""

import importlib
import math
import os
from tempfile import NamedTemporaryFile
from typing import Optional

import pyarrow.fs
import yaml
from lightning.pytorch.cli import LightningCLI
from ray import train
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer

from train.callbacks import AframeTrainReportCallback
from utils.logging import configure_logging


def get_host_cli(cli: type):
    class HostCLI(cli):
        # since this is run on the client, we don't actually
        # want to do anything with the arguments we parse,
        # just record them, so override the couple parent
        # methods responsible for actually doing stuff
        def instantiate_classes(self):
            return

        def _run_subcommand(self):
            return

        def add_arguments_to_parser(self, parser):
            """
            Add some arguments about where and how to
            run the tune job.
            """
            super().add_arguments_to_parser(parser)
            parser.add_argument("--tune.name", type=str, default="ray-tune")
            parser.add_argument(
                "--tune.space", type=str, default="train.tune.search_space"
            )
            parser.add_argument("--tune.address", type=str, default=None)
            parser.add_argument(
                "--tune.workers_per_trial", type=int, default=1
            )
            parser.add_argument("--tune.gpus_per_worker", type=int, default=1)
            parser.add_argument("--tune.cpus_per_gpu", type=int, default=8)
            parser.add_argument("--tune.num_samples", type=int, default=10)
            parser.add_argument("--tune.max_epochs", type=int, default=100)
            parser.add_argument("--tune.reduction_factor", type=int, default=4)
            parser.add_argument("--tune.storage_dir", type=str, default=None)
            parser.add_argument("--tune.min_epochs", type=int, default=1)
            parser.add_argument(
                "--tune.random_search_steps", type=int, default=10
            )

            # this argument isn't valuable for that much, but when
            # we try to deploy on local containers on LDG, the default
            # behavior will be to make a temp directory for ray cluster
            # logs at /local, which will cause permissions issues.
            parser.add_argument("--tune.temp_dir", type=str, default=None)

    return HostCLI


def get_search_space(search_space: str):
    # determine if the path is a file path or a module path
    if os.path.isfile(search_space):
        # load the module from the file
        module_name = os.path.splitext(os.path.basename(search_space))[0]
        spec = importlib.util.spec_from_file_location(
            module_name, search_space
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        # load the module using importlib
        module = importlib.import_module(search_space)

    # try to get the 'space' attribute from the module
    try:
        space = module.space
    except AttributeError:
        raise ValueError(f"Module {module.__name__} has no space dictionary")

    if not isinstance(space, dict):
        raise TypeError(
            "Expected search space in module {} to be "
            "a dictionary, found {}".format(module.__name__, type(space))
        )

    return space


def stop_on_nan(trial_id: str, result: dict) -> bool:
    return math.isnan(result["train_loss"])


def get_worker_cli(cli: LightningCLI):
    class WorkerCLI(cli):
        def instantiate_trainer(self, **kwargs):
            kwargs = kwargs | dict(
                enable_progress_bar=False,
                devices="auto",
                accelerator="auto",
                strategy=RayDDPStrategy(),
                callbacks=[AframeTrainReportCallback()],
                plugins=[RayLightningEnvironment()],
            )
            return super().instantiate_trainer(**kwargs)

    return WorkerCLI


class TrainFunc:
    """
    Callable wrapper that takes a `LightningCLI` and executes
    it with the both the `config` passed here at initialization
    time as well as the arguments supplied by a particular
    hyperparameter config. Meant for execution on workers during
    tuning run, which expect a callable that a particular
    hyperparameter config as its only argument.

    All runs of the function will be given the same
    Weights & Biases group name of `name` for tracking.
    The names of individual runs in this group will be
    randomly chosen by W&B.
    """

    def __init__(self, cli: LightningCLI, name: str, config: dict) -> None:
        self.cli = cli
        self.name = name
        self.config = config

    def __call__(self, config):
        """
        Dump the config to a file, then parse it
        along with the hyperparameter configuration
        passed here using our CLI.
        """

        with NamedTemporaryFile(mode="w") as f:
            yaml.dump(self.config, f)
            args = ["-c", f.name]
            for key, value in config.items():
                args.append(f"--{key}={value}")

            # TODO: this is technically W&B specific,
            # but if we're distributed tuning I don't
            # really know what other logger we would use
            args.append(f"--trainer.logger.group={self.name}")
            cli_cls = get_worker_cli(self.cli)
            cli = cli_cls(
                run=False, args=args, save_config_kwargs={"overwrite": True}
            )

        log_dir = cli.trainer.logger.log_dir or cli.trainer.logger.save_dir
        if not log_dir.startswith("s3://"):
            ckpt_prefix = ""
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "train.log")
            configure_logging(log_file)
        else:
            ckpt_prefix = "s3://"
            configure_logging()

        # restore from checkpoint if available
        checkpoint = train.get_checkpoint()
        ckpt_path = None
        if checkpoint:
            ckpt_path = os.path.join(
                ckpt_prefix, checkpoint.path, "checkpoint.ckpt"
            )

        # I have no idea what this `prepare_trainer`
        # ray method does but they say to do it so :shrug:
        trainer = prepare_trainer(cli.trainer)
        trainer.fit(cli.model, cli.datamodule, ckpt_path=ckpt_path)


def configure_deployment(
    train_func: TrainFunc,
    metric_name: str,
    workers_per_trial: int,
    gpus_per_worker: int,
    cpus_per_gpu: int,
    objective: str = "max",
    storage_dir: Optional[str] = None,
    fs: Optional[pyarrow.fs.FileSystem] = None,
) -> TorchTrainer:
    """
    Set up a training function that can be distributed
    among the workers in a ray cluster.

    Args:
        train_func:
            Function that each worker will execute
            with a config specifying the hyperparameter
            configuration for that trial.
        metric_name:
            Name of the metric that will be optimized
            during the hyperparameter search
        workers_per_trial:
            Number of training workers to deploy
        gpus_per_worker:
            Number of GPUs to train over within each worker
        cpus_per_gpu:
            Number of CPUs to attach to each GPU
        objective:
            `"max"` or `"min"`, indicating how the indicated
            metric ought to be optimized
        storage_dir:
            Directory to save ray checkpoints and logs
            during training.
        fs: Filesystem to use for storage
    """

    cpus_per_worker = cpus_per_gpu * gpus_per_worker
    scaling_config = ScalingConfig(
        trainer_resources={"CPU": 0},
        resources_per_worker={"CPU": cpus_per_worker, "GPU": gpus_per_worker},
        num_workers=workers_per_trial,
        use_gpu=True,
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute=metric_name,
            checkpoint_score_order=objective,
        ),
        failure_config=FailureConfig(
            max_failures=5,
        ),
        storage_filesystem=fs,
        storage_path=storage_dir,
        name=train_func.name,
        stop=stop_on_nan,
    )
    return TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
