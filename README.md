# Aframe
Detecting compact binary mergers from gravitational wave strain data using neural networks, with an emphasis on
- **Efficiency** - making effective use of accelerated hardware like GPUs in order to minimize time-to-solution
- **Scale** - validating hypotheses on large volumes of data to obtain high-confidence estimates of model performance
- **Flexibility** - modularizing functionality to expose various levels of abstraction and make implementing new ideas simple
- **Physics first** - taking advantage of the rich priors available in GW physics to build robust models and evaluate them accoring to 
meaningful metrics
- **Multi-messenger astronomy** - making algorithmic decisions and optimizations that allow for extremely low-latency alerts 

For algorithm details and performance estimates on the LVK O3 observing run, please see ["A machine-learning pipeline for real-time detection of gravitational waves from compact binary coalescences"](https://arxiv.org/abs/2403.18661). Please also cite this paper if you use `Aframe` software in your work.

## Getting Started
> **_NOTE: this repository is a WIP. Please open up an issue if you encounter bugs, quirks, or any undesired behavior_**

> **_NOTE: Running Aframe out-of-the-box requires access to an enterprise-grade GPU (e.g. P100, V100, T4, A[30,40,100], etc.). There are several nodes on the LIGO Data Grid which meet these requirements_**.

Please see the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) for help on setting up your environment 
on the [LIGO Data Grid](https://computing.docs.ligo.org/guide/computing-centres/ldg/) (LDG) and for configuring access to [Weights and Biases](https://wandb.ai), and the [Nautilus hypercluster](https://ucsd-prp.gitlab.io/). 
This quickstart includes a Makefile and instructions for setting up all of the necessary software, environment variables, and credentials 
required to run `Aframe`. 

Once setup, create a [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of this repository, and clone it

```bash
git clone git@github.com:albert-einstein/aframe.git
```

`Aframe` utilizes `git` [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Make sure to initialize and update those

``bash
git submodule update --init
```

Now, you should be all setup! The default Aframe experiment is the [`sandbox`](./aframe/pipelines/sandbox/) pipeline found under `aframe/pipelines/sandbox`. It is recommended that you follow the instructions in the sandbox [README](./aframe/pipelines/sandbox/) and execute the pipeline as an introduction to interacting with the respository. 


## Repository structure
The code here is structured like a [monorepo](https://medium.com/opendoor-labs/our-python-monorepo-d34028f2b6fa), with applications siloed off into isolated environments to keep dependencies lightweight, but built on top of a shared set of libraries to keep the code modular and consistent. Briefly, the repository consists of three main sub-directories:

1. [projects](./projects/README.md) - containerized sub-tasks of aframe analysis that each produce _artifacts_
2. [libs](./libs/README.md) - general-purpose functions (libraries) mean to support more than one project
3. [aframe](./aframe/README.md) - `law` wrappers for building complex pipelines of project tasks.

For more details on each of these, please see their respective README's. 

## Contributing
If you are looking to contribute to `Aframe`, please see our [contribution guidelines](./CONTRIBUTING.md)


### One layer up: `luigi` and `law`
That command above is simple enough, but it might be nice to 1) specify arguments with configs, and 2) Incorporate tasks as steps in a larger pipeline.
To do this, this repo takes advantage of a library called `luigi` (and a slightly higher-level wrapper, `law`) to construct configurable, modular tasks that can be strung into pipelines. 
To understand the structure of `luigi` tasks, it is reccommended to read the [docs](https://luigi.readthedocs.io/en/stable/).

The top level `aframev2` repository contains the [environment](pyproject.toml) that is used to launch tasks with `luigi` and `law`.
to install this environment, simply run 

```
poetry install
```

in the root of this repository.

To run a local training job you can now run 

```
poetry run law run aframe.TrainLocal \
    --gpus <ID of GPUs to train on> \
    --image ~/aframe/images/train.sif \
    --config /opt/aframe/projects/train/config.yaml \
    --data-dir ~/aframe/data/train \
    --run-dir ~/aframe/results/my-first-luigi-run \
    --use-wandb \
    --wandb-name my-first-luigi-run
```
This has taken care of setting some sensible defaults for you, and allows for simpler syntax like the `--gpus` arg and `--use-wandb` which will configure most of your W&B settings for you.
All tasks also come with a built-in `--dev` arg which will automatically map your current code into the container for super low-friction development.

To see all the parameters a task has to offer, you can run e.g.
```
poetry run law run aframe.tasks.TrainLocal --help
```

### Final Layer: Pipelines
As mentioned, `luigi` and `law` allow for the construction of large scale pipelines. The `aframe/pipelines/` directory contains common analysis pipelines. 
Currently, only the `sandbox` pipeline is available. This pipeline will launch a single end-to-end `aframe` workflow consisting of training / testing data generation, model training, model export, and inference using a triton server. The easiest way to run the pipeline is to use the config file, which is specified in `law` via the `LAW_CONFIG_FILE` environment variable:

```
LAW_CONFIG_FILE=aframe/pipelines/sandbox/sandbox.cfg poetry run law run aframe.pipelines.sandbox.Sandbox --gpus <GPU IDs> 
```

### One more layer: local hyperparameter tuning
To search over hyperparameters, you can launch a local hyperparameter tuning job by running

```
APPTAINERENV_CUDA_VISIBLE_DEVICES=<IDs of GPUs to tune on> apptainer run --nv --bind .:/opt/aframe ~/aframe/images/rain.sif \
    python -m train.tune \
        --config /opt/aframe/projects/train/config.yaml
        --data.ifos=[H1,L1]
        --data.data_dir ~/aframe/data/train
        --trainer.logger=WandbLogger
        --trainer.logger.project=aframe
        --trainer.logger.save_dir=~/aframe/results/my-first-tune \
        --tune.name my-first-tune \
        --tune.storage_dir ~/aframe/results/my-first-tune \
        --tune.temp_dir ~/aframe/results/my-first-tune/ray \
        --tune.num_samples 8 \
        --tune.cpus_per_gpu 6 \
        --tune.gpus_per_worker 1 \
        --tune.num_workers 4
```
This will launch 8 hyperparameter search jobs that will execute on 4 workers using the Asynchronous Successive Halving Algorithm (ASHA).
All the runs will be given the same **group** ID in W&B, and will be assigned random names in that group.

**NOTE: for some reason, right now this will launch one job at a time that takes all available GPUs. This needs sorting out**

The cool thing is that if you already have a ray cluster running somewhere, you can distribute your jobs over that cluster by simply adding the `--tune.endpoint <ip address of ray cluster>:10001` command line argument.

This isn't implemented at the `luigi` level yet, but the skeleton of how it will work is in `aframe/tasks/train.py`.

### TODO: discussion about moving to remote and higher-friction production workloads

### Useful concepts and things to be aware of
- Weights & Biases
    - You can assign various attributes to your W&B logger
        - `name`: name the run will be assigned
        - `group`: group to which the run will be assigned. This is useful for runs that are part of the same experiment but execute in different scripts, e.g. a hyperparameter sweep or maybe separate train, inferenence, and evaluation scripts
        - `tags`: comma separate list of tags to give your run. Makes it easy to filter in the dashboard e.g. for `autoencoder` runs
        - `project`: the workspace consisting of multiple related experiments that your run is a part of, e.g. `aframe`
        - `entity`: the group managing the experiments your run is associated, e.g. `ml4gw`. If left blank, the project and run will be associated with your personal account
- Tuning
    - You don't need to specify the `temp_dir` when tuning remotely, this is just a consequence of `ray` trying to write to a root directory for temp files that breaks on LDG
    - If you're tuning remotely, your `storage_dir` should be a remote S3 bucket that all your workers can access. You'll need to specify an `AWS_ENDPOINT_URL` environment variable for those workers so they know where your bucket lives

## Where things stand and where they can go
### Optimization schemes
- Semi-supervised
    - Simple autoencoder implementation working, using max correlation across shifts as loss function
        - See comments in model subclass for discussion on how to make this more exotic
    - The key to this approach is that neither interferometer's prediction about the other is allowed to depend on any _info_ about the other
        - Grouped convolutions ensure this for convolutional autoencoder architecture
        - Will be more complex for more than 2 IFOs
    - Requires using `ml4gw` branch with autoencoder library
    - One potential future direction could be to build on Deep's VICReg work and enforce that
        - Representations from the same event are similar in both IFOs
            - While you don't want one channel's _prediction_ to depend on the other channel at all, there's nothing wrong with imposing a _loss_ that combines their information
        - You could even do 2 sky samplings for the same event and enforce that these are similar
        - This would involve using the model-specific loss terms discussed in the comments under the model
### Deployment scenarios
- LDG local
    - This is working, though see my note about local tunings above
- Remote via Kubernetes
    - The tuning script should work if a cluster is already up, and training is trivial to run with a kubernetes deploy yaml
    - There's a class in `aframe/base.py` that uses ray-kube to spin up a cluster in a law Task
        - Not tested yet, but should be basically what you're looking for
        - Use the `configure_cluster` method to do task-specific cluster configuration before launching, e.g. setting secrets, environment variables, etc.
        - Needs ray-kube to implement [this functionality](https://github.com/EthanMarx/ray-kube/issues/5) so that we can e.g. set `AWS_ENDPOINT_URL` to the desired target
            - Somewhat silly, but will probably be helpful to automatically map to the internal S3 endpoint based on the external endpoint, that way users only need to specify one that will work in both local and remote scenarios
### Scales
- Data-distributed model training
    - Implemented, just expose multiple GPUs to your training job
-  Distributed hyperparameter searching
    - See issues with local deployment discussed above
    - Remote works, but not luigi-fied yet. See discussion in [Deployment Scenarios](#Deployment-scenarios)

## What else?
There's tons of `TODOS` littering the code that cover stuff I'll have missed here.
One major one is the ability to log plots of model predictions to W&B during validation. See my comments on it [here](https://github.com/ML4GW/aframev2/blob/b2a5164d2e49f9c2701e2100091f6b9b8467678a/projects/train/train/model/base.py#L74-L87).
Basically you should be able to define callbacks for various tasks that have an `on_validation_score` method to pass model inputs and outputs that you can log to W&B.
I think this will be particularly important for the autoencoder work, where visualizing what it's learning will be instructive.

More broadly, it will be useful to start up-leveling some of the training framework utilities to library that sits one level above `ml4gw`. I'm thinking of
- Simple functionality for logging plots iteratively to W&B during training in a single table (see how I do this in [deepclean](https://github.com/alecgunny/deepclean-demo/blob/0874cb91e35b4bba0a3b62dfd33dae267c2deff7/projects/train/train/callbacks.py#L12))
- Exporting of the torch trace at the end of training
- The tuning library, which is actually pretty general already
- Some of the data access and distribution utilities built into the base `Dataset` here
- And of course all of the `luigi`/`law` stuff, which would probably be its own library even one layer above this.
