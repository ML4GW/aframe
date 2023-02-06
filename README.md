# BBHNet
Detecting binary blackhole mergers from gravitational wave strain timeseries data using neural networks, with an emphasis
- **Efficiency** - making effective use of accelerated hardware like GPUs in order to minimize time-to-solution.
- **Scale** - validating hypotheses on large volumes of data to obtain high-confidence estimates of model performance
- **Flexibility** - modularizing functionality to expose various levels of abstraction and make implementing new ideas simple
- **Physics first** - taking advantage of the rich priors available in GW physics to build robust models and evaluate them accoring to meaningful metrics

BBHNet represents a _framework_ for optimizing neural networks for detection of CBC events from time-domain strain, rather than any particular network architecture.

## Quickstart
> **_NOTE:_** right now, BBHNet can only be run by LIGO members

> **_NOTE:_** Running BBHNet out-of-the-box requires access to an enterprise-grade GPU (e.g. P100, V100, T4, A[30,40,100], etc.). There are several nodes on the LIGO Data Grid which meet these requirements.

### 1. Setting up your environment for data access
In order to access the LIGO data services required to run BBHNet, start by following the instructions [here](https://computing.docs.ligo.org/guide/auth/kerberos/#usage) to set up a kerberos keytab for passwordless authentication to LIGO data services

```console
$ ktutil
ktutil:  addent -password -p albert.einstein@LIGO.ORG -k 1 -e aes256-cts-hmac-sha1-96
Password for albert.einstein@LIGO.ORG:
ktutil:  wkt ligo.org.keytab
ktutil:  quit
```
with `albert.einstein` replaced with your LIGO username. Move this keytab file to `~/.kerberos`

```console
mkdir ~/.kerberos
mv ligo.org.keytab ~/.kerberos
```
You'll also want to create directories for storing X509 credentials, input data, and BBHNet outputs.

```console
mkdir -p ~/cilogon_cert ~/bbhnet/data ~/bbhnet/results
```

### 2. Install `pinto`
BBHNet leverages both Conda and Poetry to manage the environments of its projects. For this reason, end-to-end execution of the BBHNet pipeline relies on the [`pinto`](https://ml4gw.gitub.io) command line utility. Please see the [Conda-based installation instructions](https://ml4gw.github.io/pinto/#conda) for `pinto` in its documentation and continue once you have it installed. You can confirm your installation by running

```console
pinto --version
```

### 3. Run the `sandbox` pipeline
The default BBHNet experiment is the [`sandbox`](./projects/sandbox) pipeline found under the `projects` directory. If you're on a GPU-enabled node on the LIGO Data Grid (LDG) and have completed the steps above, start by defining a couple environment variables

```console
# BASE_DIR is where we'll write all logs, training checkpoints,
# and inference/analysis outputs. This should be unique to
# each experiment you run
export BASE_DIR=~/bbhnet/results/my-first-run

# DATA_DIR is where we'll write all training/testing
# input data, which can be reused between experiment
# runs. Just be sure to delete existing data or use
# a new directory if a new experiment changes anything
# about how data is generated, because BBHNet by default
# will opt to use cached data if it exists.
export DATA_DIR=~/bbhnet/data
```

then from the `projects/sandbox` directory, just run

```console
BASE_DIR=$BASE_DIR DATA_DIR=$DATA_DIR pinto run
```

This will execute training and inference pipeline which will:
- Download background and glitch datasets and generate a dataset of raw gravitational waveforms
- Train a 1D ResNet architecture on this data
- Accelerate the trained model using TensorRT and export it for as-a-service inference
- Serve up this model with Triton Inference Server via Singularity, and use it to run inference on a dataset of timeshifted background and waveform-injected strain data
- Use these predictions to generate background and foreground event distributions
- Serve up an application for visualizing and analyzing those distributions at `localhost:5005`.

Note that the first execution may take a bit longer than subsequent runs, since `pinto` will build all the necessary environments at run time if they don't already exist. The environments for data generation and training in particular can be expensive to build because the former is built with Conda and the latter requires building GPU libraries.

### 3b. Simplify with a `.env`
Since `pinto` supports using `.env` files to specify environment variables, consider creating a `projects/sandbox/.env` file and specifying `BASE_DIR` and `DATA_DIR` there:

```bash
BASE_DIR=$HOME/bbhnet/results/my-first-run
DATA_DIR=$HOME/bbhnet/data
```

Then you can simplify the above expression to just
```console
pinto run
```

Another useful way to set things up is to write `projects/sandbox/.env` like
```bash
BASE_DIR=$HOME/bbhnet/results/$PROJECT
DATA_DIR=$HOME/bbhnet/data
```

then redefine the `PROJECT` environment variable for each new experiment you run so that it's given its own results directory, e.g.

```console
PROJECT=my-second-run pinto run
```

`pinto` will automatically pick up the local `.env` file and fill in the `$PROJECT` variable with the value set at the command line.

## Experiment overview
### Binary blackhole detection with deep learning
The gravitational wave signatures generated by the merger of binary blackhole (BBH) systems are well understood by general relativity, and powerful models for simulating these waveforms are easily accessible with modern GW physics software libraries. These simulated waveforms (or more accurately their frequency-domain representations) can then be used for matched-filter searches. This represents the most common existing method for detecting BBH events.

Matched filtering in this context, however, has its limitations.
- The number of parameters of the BBH system, which conditions the waveform generated by its merger, is sufficiently large that matched filter template banks must contain on the order of millions of templates for high-sensitivity searches. This makes real-time searches computationally intensive.
- Un-modelled non-Gaussian artifacts in the interferometer background, or **glitches**, can reduce the sensitivity of matched filters. Though most searches implement veto mechanisms to mitigate the impact of these glitches, they remain a persistent source of false alarms.

Deep learning algorithms represent an attractive alternative to these methods because they can "bake-in" the cost of evaluating large template banks up-front during training, trading this for efficient inference evaluation at run-time and drastically reducing the compute resources required for online searches. Moreover, the same simulation methods that enable matched filtering also allow for generation of arbitrary volumes of training data that can be used to fit robust models of the signal space.

While glitches can also represent problematic inputs for neural networks, they offer the potential to learn to exclude them by sheer "brute-force": providing networks with lots of examples of glitches during training in order to learn to distinguish them from real events.

BBHNet attempts to apply deep learning methods to this problem by combining these observations and leveraging both the powerful existing models of BBH signals and the enormous amount of existing data collected by LIGO to build robust datasets of both background and signal-containing samples. More specifially, we:
- Use the `ml4gw` library to project a dataset of pre-computed gravitational waveforms to interferometer responses on-the-fly on the GPU. This allows us to efficiently augment our dataset of signals by "observing" the same event at any point on the celestial sphere and at arbitrary distances (the latter achieved by remapping its SNR relative to the background PSD of the training set. Note that this will by extension change the observed mass of the blackholes in the _detector frame_)
- Use the `pyomicron` utility to search through the training set (and periods before it) for glitches which we can oversample during training and randomly use to replace each interferometer channel independently.

### Evaluating the performance of a trained network
TODO: fill this out or just refer to the documentation of `infer`.


## Development instructions.
By default, `pinto` uses Poetry to install all local libraries editably. This means that changes you make to your local code will automatically be reflected in the libraries used at run time. For information on how to help your new code best fit the structure of this repository, see the [contribution guidelines](./CONTRIBUTING.md).

### Code Structure
The code here is structured like a [monorepo](https://medium.com/opendoor-labs/our-python-monorepo-d34028f2b6fa), with applications siloed off into isolated environments to keep dependencies lightweight, but built on top of a shared set of libraries to keep the code modular and consistent.

Note that this means there is no "BBHNet environment:" you won't find an `environment.yaml` or poetry config at this root level. Each project is associated with its own environment which is defined _locally with respect to the project itself_. For instructions on installing each project, see its associated documentation (though in general, running `pinto build` from the project's directory will be sufficient).

If you run the pipeline using the [instructions above](#3.-run-the-`sandbox`-pipeline), the environment associated with each step in the pipeline (i.e. each child project's environment) will be built before running the step if it does not already exist. This is true of running `pinto run` for each step individually as well.

Note as well that each project is associated with one or multiple scripts or applications, which are defined in the `[tool.poetry.scripts]` table of the project's `pyproject.toml`. These scripts will be able to be executed as command-line executables inside the project's environment, e.g.

```console
# run in projects/sandbox/train
pinto run train -h
```

Some projects, such as `projects/sandbox/datagen` will be associated with several scripts that perform different functionality using the same environment, e.g.

```console
# run these in projects/sandbox/datagen
pinto run generate-background -h
pinto run generate-glitches -h
pinto run generate-timeslides -h
```

## Tips and tricks
Given the more advanced structure of the repo outlined above, there are some best practices you can adopt while developing which can make your life easier and your code simpler to get integrated:
- Most scripts within projects parse their commands using the [`typeo`](https://github.com/ml4gw/typeo) utility, which will automatically create a command-line parser using the arguments and associated type annotations of a function and execute this parser when the function is called with no arguments. This means you can execute scripts by passing the arguments of the associated function explicitly:
```console
pinto run train --learning-rate 1e-3 ... resnet --layers 2 2 2 2  ...
```
or by pointing to a `pyproject.toml` (or directory containing a `pyproject.toml`) which defines all the relevant arguments
```console
# this says "run the train command, but parse the arguments
# for it from [tool.poetry.scripts.train] table of the
# pyproject.toml contained in the directory directly above this
# (signified by the ..) using the resnet subcommand (a sub-table
# of the [tool.poetry.scripts.train] table)"
pinto run train --typeo ..:train:resnet
```
For information on how to read a `typeo` config, see its [README](https://github.com/ml4gw/typeo/tree/main/README.md).
- Make aggressive use of branching (`git chekout -b debug-some-minor-issue`), even from development branches (i.e. not `main`). This will ensure that good ideas don't get lost in the process of debugging, and that your main development branch remains as stable as possible. Once you've solved the issue you branched out to fix, you can `git checkout` back to your main development branch, `git merge debug-some-minor-issue` the fix in, then delete the temporary branch `git branch -d debug-some-minor-issue`
- When you want to pull the latest changes in from `upstream main` to fork off a new development branch, consider using `git pull --rebase upstream main`. This will ensure that `git pull` doesn't create an extraneous merge commit that starts to diverge the histories of your local `main` branch and the upstream `main` branch, which can make future pull requests harder to scrutinize.
- `pinto` is, at its core, a pretty thin wrapper around `conda` and `poetry`. If you're experiencing any issues with your environments, try running the relevant `conda` or `poetry` commands explictly to help debug. If things seem truly hopeless, delete the environment entirely (e.g. `rm -rf ~/minicondae3/envs/<env-name>`) and rebuild it using the right combination of conda and poetry commands (`conda env create -f ...` followed by a `conda activate` and `poetry install` for conda-managed projects and just plain old `poetry install` for poetry-managed projects).
