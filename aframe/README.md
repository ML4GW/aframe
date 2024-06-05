# Aframe Tasks
[`luigi`](https://github.com/spotify/luigi) and [`law`](https://github.com/riga/law) `Tasks` for building scalable, complex, and dynamic `Aframe` analyses

> **Note** It is expected that you are familiar with `Aframe` [projects](../projects/README.md) and have built their container images

> **Note** It is highly recommended that you read the luig [docs](https://luigi.readthedocs.io/en/stable/) to understand the structure of `luigi` tasks! Also, looking at `law` [examples](https://github.com/riga/law/tree/master/examples/), specifically the htcondor workflows, will be useful


## Environment setup
The top level `aframe` repository contains the [environment](pyproject.toml) that is used to launch `Tasks` with `law`

In the root of this repository, simply run 

```
poetry install
```

to install this environment


A number of these `Tasks` launch jobs using `condor`, with the accounting group and username inferred from environment variables `LIGO_GROUP` and `LIGO_USERNAME`, respectively. These variables can be specified when launching the pipeline in the command line, or they can be added to your `.bash_profile` (or other relevant shell configuration file) with:

```bash
echo export LIGO_USERNAME=<your albert.einstein username> >> ~/.bash_profile
echo export LIGO_GROUP=ligo.dev.o4.cbc.allsky.aframe >> ~/.bash_profile
```
  

## Introduction to Tasks
Beyond running individual projects in containerized environments, it might be nice to 1) specify arguments with configs, and 2) incorporate tasks as steps in a larger pipeline. To do this, we take advantage of a library called `luigi` (and a slightly higher-level wrapper, `law`) to construct configurable, modular, scalable tasks that can be strung into pipelines. 


`Tasks` are isolated pieces of code that are meant to run in isolated `Aframe` [project](../projects/README.md) environments. 

A `Task` is a class that defines two main methods 

- `requires()` specifies other `Task` dependencies
- `output()` specifies a `Target`, which typically corresponds to an artifact on disk

A `Task` is considered complete if it's `output` "exists". Complete `Tasks` will not be re-run.

`Tasks` can be run by specifying the python path to the `Task`, e.g.
```
poetry run law run aframe.tasks.TrainLocal ...
```

## Tips and Tricks
All `Aframe` `Tasks` also come with a built-in `--dev` arg which will automatically map your current code into the container using Apptainers `--bind` for low-friction development.


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
