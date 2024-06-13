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

Next, in the root of the repository, install the base `aframe` `poetry` environment.
```
poetry install
```
This environment is used to launch `luigi`/`law` tasks and pipelines. See this [README](./aframe) for more information.

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
