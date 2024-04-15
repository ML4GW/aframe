# aframe
[`luigi`](https://github.com/spotify/luigi) and [`law`](https://github.com/riga/law) `Tasks` for building scalable, complex, and dynamic `Aframe` analyses

> **Note** It is expected that you are familiar with `Aframe` [projects](../projects/README.md) and have built their container images

## Introduction
Beyond running individual projects in containerized environments, it might be nice to 1) specify arguments with configs, and 2) incorporate tasks as steps in a larger pipeline.
To do this, we take advantage of a library called `luigi` (and a slightly higher-level wrapper, `law`) to construct configurable, modular tasks that can be strung into pipelines. To understand the structure of `luigi` tasks, it is recommended to read the [docs](https://luigi.readthedocs.io/en/stable/).

 
## Environment setup
The top level `aframe` repository contains the [environment](pyproject.toml) that is used to launch `Tasks` with `luigi` and `law`.
To install this environment, simply run 

```
poetry install
```

in the root of this repository.  

## Tasks
In short, `Tasks` are isolated scripts that are run in containerized `Aframe` [project](../projects/README.md) environments.

A `Task` is a class that defines two main methods 

- `requires()` specifies other `Task` dependencies
- `output()` specifies a `Target`, which typically corresponds to a artifact on disk

A `Task` is considered complete if it's `output` "exists". Complete `Tasks` will not be re-run.

All `Aframe` `Tasks` also come with a built-in `--dev` arg which will automatically map your current code into the container using `--bind` for super low-friction development.

`Tasks` can be run by specifying the python path to the `Task`, e.g.
```
poetry run law run aframe.tasks.TrainLocal ...
```

to see potential arguments for a `Task` you can run e.g.

```
poetry run law run aframe.tasks.TrainLocal --help
```


## Examples

### Local Training

```
poetry run law run aframe.Train \
    --gpus <ID of GPUs to train on> \
    --image ~/aframe/images/train.sif \
    --config /opt/aframe/projects/train/config.yaml \
    --data-dir ~/aframe/data/train \
    --run-dir ~/aframe/results/my-first-luigi-run \
    --use-wandb \
    --wandb-name my-first-luigi-run
```

This has taken care of setting some sensible defaults for you, and allows for simpler syntax like the `--gpus` arg and `--use-wandb` which will configure most of your W&B settings for you.

### Remote Training

```
poetry run law run aframe.Train \
    --gpus <ID of GPUs to train on> \
    --image ~/aframe/images/train.sif \
    --config /opt/aframe/projects/train/config.yaml \
    --data-dir ~/aframe/data/train \
    --run-dir ~/aframe/results/my-first-luigi-run \
    --use-wandb \
    --wandb-name my-first-luigi-run
    --train-remote
```
