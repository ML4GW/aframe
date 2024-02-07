# aframev2
An attempt to overhaul and modernize the infrastructure to run [`aframe`](https://github.com/ML4GW/aframe). Unifying multiple threads of research
### Model architecture classes
- Time domain
- Frequency Domain
### Optimization schemes
- Supervised
- Semi-supervised
### Deployment scenarios
- LIGO Data Grid (LDG) local
- Remote via Kubernetes
### Scales
- Data-distributed model training
- Distributed hyperparameter searching

## Getting Started
Please see the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) for help on setting up your environment 
on the [LIGO Data Grid](https://computing.docs.ligo.org/guide/computing-centres/ldg/) (LDG) and for configuring access to [Weights and Biases](https://wandb.ai), and the [Nautilus hypercluster](https://ucsd-prp.gitlab.io/). 
This quickstart includes a Makefile and instructions for setting up all of the necessary software, environment variables, and credentials 
required to run `aframe`. 

**NOTE: this repository is a WIP. You will encounter bugs, quirks, and undesired behavior. If you have any suggestions on making the development process easier, please open up an issue!**

### Quickstart: low-friction, local development
Each sub-task in `aframe` is implemented as a containerized application, whose environment and Apptainer [definition file](https://apptainer.org/docs/user/1.0/definition_files.html) live with the code they're meant to deploy. These live under the `projects` sub-directory. The projects include

- `data` : Querying strain data and generating waveforms for training and testing.
- `train` : Pytorch lightning code for training neural-networks.
- `export`: Exporting trained networks as accelerated executables for inference. 
- `infer`: Launching triton inference servers and deploying inference clients to analyze timeslides and injections.
- `utils`: General utilites used by all projects (TODO: move this under `libs`)

You can build and execute code inside these containers locally. As an example, let's go through the process of generating data for training `aframe`. 
First, you will need to build the `data` project container:

```
mkdir ~/aframe/images
cd projects/data
apptainer build ~/aframe/images/data.sif apptainer.def
```

Once that is complete, let's query for open science segments containing high-quality data:

```
mkdir ~/aframe/data/
apptainer run ~/aframe/images/data.sif \
    python -m data query --flags='["H1_DATA", "L1_DATA"]' --start 1240579783 --end 1241443783 --output_file ~/aframe/data/segments.txt
```

Inspecting the output, it looks like theres quality data segments  `(1240579783, 1240579783)` and `(1240594562, 1240606748)`. Let's fetch strain data during those segments. One will be used for training, the 

```
apptainer run ~/aframe/images/data.sif \
    python -m data fetch \
    --start 1240579783 \
    --end 1240579783 \
    --channels='["H1", "L1"]' \ 
    --sample_rate 2048 \
    --output_directory ~/aframe/data/background/

apptainer run ~/aframe/images/data.sif \
    python -m data fetch \
    --start 1240594562 \
    --end 1240606748 \
    --channels='["H1", "L1"]' \ 
    --sample_rate 2048 \
    --output_directory ~/aframe/data/background/
```

Finally, lets generate some waveforms:

```
apptainer run ~/aframe/images/data.sif \
    python -m data waveforms \
    --prior data.priors.priors.end_o3_ratesandpops \
    --num_signals 10000 \
    --waveform_duration 8 \
    --sample_rate 2048 \
    --output_file ~/aframe/data/signals.hdf5
```

Great! We are now ready to train a model! In the same fashion, let's build the training container:

```
mkdir ~/aframe/images
cd projects/train
apptainer build ~/aframe/images/train.sif apptainer.def
```

and launch a training job!
```
mkdir ~/aframe/results
APPTAINERENV_CUDA_VISIBLE_DEVICES=<ID of GPU you want to train on> apptainer run --nv ~/aframe/images/train.sif \
    python -m train \
        --config /opt/aframe/projects/train/config.yaml \
        --data.ifos=[H1,L1] \
        --data.data_dir ~/aframe/data/train \
        --trainer.logger=WandbLogger \
        --trainer.logger.project=aframe \
        --trainer.logger.name=my-first-run \
        --trainer.logger.save_dir=~/aframe/results/my-first-run
```

This will infer most of your training arguments from the YAML config that got put into the container at build time. If you want to change this config, or if you change any code and you want to see those changes reflected inside the container, you can simply update the start of the command to read `apptainer run --nv --bind .:/opt/aframe`. 

Once your run is started, you can go to [wandb.ai](https://wandb.ai) and track your loss and validation score. If you don't want to track your run with W&B, just remove all the first three `--trainer` arguments above. This will save your training metrics to a local CSV in the `save_dir`.

You can even train using multiple GPUS, simply by specifying a list of comma-separated GPU indices to `APPTAINERENV_CUDA_VISIBLE_DEVICES`.

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
