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

## How to get started
All the instructions you'll see here are intended for running on LDG. Start by ensuring you have:
- A local installation of `poetry` (TODO: link)
- LIGO data credentials (TODO: standard link for this)
- A Nautilus cluster account (TODO: link)
- S3 storage credentials (TODO: link to Nautilus instructions) and that they're written to a local file `$HOME/.aws/credentials` as
```
[default]
aws_access_key_id = <access key>
aws_secret_access_key = <secret key>
```
- A Weight & Biases account for remote experiment tracking

### Quickstart: low-friction, local development
Each sub-task in `aframe` is implemented as a containerized application, whose environment and Apptainer [definition file](https://apptainer.org/docs/user/1.0/definition_files.html) live with the code they're meant to deploy.
You can build and execute code inside these containers locally. For example, to build the training application:

```
mkdir ~/aframe/images
cd projects/train
apptainer build ~/aframe/images/train.sif apptainer.def
```

This will build an Apptainer container image at `~/aframe/images/train.sif`. You can then launch a local training run (assuming you're on a node with a GPU) by doing something like (assuming you've already generated some data TODO: add this step)

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

### One layer up: `luigi`
That command above is simple enough, but it might be nice to 1) specify e.g. W&B arguments with configs, and 2) longer term, incorporate this train task as one step in a larger pipeline.
To do this, this repo takes advantage of a library called `luigi` (and a slightly higher-level wrapper, `law`) to construct configurable, modular tasks that can be strung into pipelines.
To execute the train task via `luigi`, first install the current project via `poetry`

```
poetry install
```

Then run

```
poetry run law run aframe.TrainLocal \
    --gpus <ID of GPU to train on> \
    --image ~/aframe/images/train.sif \
    --config /opt/aframe/projects/train/config.yaml \
    --data-dir ~/aframe/data/train \
    --run-dir ~/aframe/results/my-first-luigi-run \
    --use-wandb \
    --wandb-name my-first-luigi-run
```
This has taken care of setting some sensible defaults for you, and allows for simpler syntax like the `--gpus` arg and `--use-wandb` which will configure most of your W&B settings for you.
All tasks also come with a built-in `--dev` arg which will automatically map your current code into the container for super low-friction development.

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
### Model architecture classes
- Time Domain
    - Fully implemented and working for supervised case
- Frequency Domain
    - Not implemented yet, but all the classes will look almost exactly the way they look for the time domain supervised case, except for
        - `augment` on the `Dataset` object should just call `super().augment(X)` then `return spectrogram(X)`. This will require adding `torchaudio` as a dependency
        - The `Architecture` subclass will just need to implement a 2D ResNet from `torchvision`. See how the time domain supervised case wraps `ml4gw`'s 1D ResNet
### Optimization schemes
- Supervised
    - Implemented (for time domain, see frequency domain discussion above)
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
