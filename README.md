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
> **_Note: this repository is a WIP. Please open up an issue if you encounter bugs, quirks, or any undesired behavior_**

> **_Note: Running Aframe out-of-the-box requires access to an enterprise-grade GPU(s) (e.g. P100, V100, T4, A[30,40,100], etc.). There are several nodes on the LIGO Data Grid which meet these requirements_**.

Please see the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) for help on setting up your environment 
on the [LIGO Data Grid](https://computing.docs.ligo.org/guide/computing-centres/ldg/) (LDG) and for configuring access to [Weights and Biases](https://wandb.ai), and the [Nautilus hypercluster](https://ucsd-prp.gitlab.io/). 
This quickstart includes a Makefile and instructions for setting up all of the necessary software, environment variables, and credentials 
required to run `Aframe`. 

Once setup, create a [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of this repository, and clone it.

> **_Note: Ensure that you have added a [github ssh key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to your account_**

```bash
git clone git@github.com:albert-einstein/aframev2.git
```

Set the `AFRAME_REPO` environment variable in your `~/.bash_profile` to point to the location where you cloned the repository.

`Aframe` utilizes `git` [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Make sure to initialize and update those

```bash
git submodule update --init
```

When pulling changes from this repository, it's recommended to use the `--recurse-submodules` flag to pull any updates from the submodules as well.


Next, follow the [instructions](./projects/README.md) for building each project's Apptainer images, and familiarize yourself with the various projects. These images are used as environments for running `Aframe` workflows, and thus are necessary to build. Once complete, you are all setup! 

The default Aframe experiment is the [`sandbox`](./aframe/pipelines/sandbox/) pipeline found under `aframe/pipelines/sandbox`. First follow the general instructions in the aframe [README](./aframe), and then the instructions in the sandbox [README](./aframe/pipelines/sandbox/) to execute the pipeline as an introduction to interacting with the respository. 


## Repository structure
The code here is structured like a [monorepo](https://medium.com/opendoor-labs/our-python-monorepo-d34028f2b6fa), with applications siloed off into isolated environments to keep dependencies lightweight, but built on top of a shared set of libraries to keep the code modular and consistent. Briefly, the repository consists of three main sub-directories:

1. [projects](./projects/README.md) - containerized sub-tasks of aframe analysis that each produce _artifacts_
2. [libs](./libs/README.md) - general-purpose functions (libraries) mean to support more than one project
3. [aframe](./aframe/README.md) - `law` wrappers for building complex pipelines of project tasks.

For more details, please see the respective README's. 

## Contributing
If you are looking to contribute to `Aframe`, please see our [contribution guidelines](./CONTRIBUTING.md)


### TODO: Move the below to corresponding documentation location / open up issues
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

## TODO's
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
