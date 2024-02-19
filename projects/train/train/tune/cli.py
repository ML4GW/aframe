import os
from typing import Optional

import pyarrow.fs
import ray
import s3fs
import yaml
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from train.cli import AframeCLI
from train.tune import utils as tune_utils


def main(args: Optional[list[str]] = None):
    host_cli = tune_utils.get_host_cli(AframeCLI)

    # create a yaml dict version of whatever arguments
    # we passed at the command line to pass again in
    # each train job
    cli = host_cli(run=False, args=args)
    config = cli.parser.dump(cli.config, format="yaml")
    config = yaml.safe_load(config)

    # pop out the arguments specific to the tuning
    # and initialize a session if there's any existing
    # cluster we should connect to
    tune_config = config.pop("tune")
    if "address" in tune_config:
        address = "ray://" + tune_config.pop("address")
    else:
        address = None
    ray.init(address, _temp_dir=tune_config.get("temp_dir", None))

    # directly use s3 instead of rays pyarrow  s3 default due to
    # this issue https://github.com/ray-project/ray/issues/41137
    fs = None
    storage_dir = tune_config.get("storage_dir", "")
    if storage_dir.startswith("s3://"):
        storage_dir = storage_dir.removeprefix("s3://")

        fs = s3fs.S3FileSystem(
            key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        )
        fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs))

    # construct the function that will actually
    # execute the training loop, and then set it
    # up for Ray to distribute it over our cluster,
    # with the desired number of resources allocated
    # to each running version of the job
    train_func = tune_utils.configure_deployment(
        tune_utils.TrainFunc(AframeCLI, tune_config["name"], config),
        metric_name="valid_auroc",
        workers_per_trial=tune_config["workers_per_trial"],
        gpus_per_worker=tune_config["gpus_per_worker"],
        cpus_per_gpu=tune_config["cpus_per_gpu"],
        objective="max",
        storage_dir=storage_dir or None,
        fs=fs,
    )
    scheduler = ASHAScheduler(
        max_t=tune_config["max_epochs"],
        grace_period=tune_config["min_epochs"],
        reduction_factor=tune_config["reduction_factor"],
    )

    search_space = tune_utils.get_search_space(tune_config["space"])

    # restore from a previous tuning run
    path = os.path.join(storage_dir, tune_config["name"])
    if tune.Tuner.can_restore(path):
        tuner = tune.Tuner.restore(
            path, train_func, resume_errored=True, storage_filesystem=fs
        )

    else:
        tuner = tune.Tuner(
            train_func,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="valid_auroc",
                mode="max",
                num_samples=tune_config["num_samples"],
                scheduler=scheduler,
                reuse_actors=True,
            ),
        )
    tuner.fit()


if __name__ == "__main__":
    main()
