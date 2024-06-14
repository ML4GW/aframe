# Pipelines

> **_Note: It is highly recommended that you have completed the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) instructions before running the sandbox pipeline_**

> **_Note: It is assumed that you have already built each [project's container](../../../projects/README.md)_**

Pipelines string together `luigi` / `law` tasks to run an end-to-end generic aframe workflow.
In short, pipelines will

1. Generate training data 
2. Generate testing data
3. Train or Tune a model
4. Export trained weights to TensorRT
5. Perform inference using Triton
6. Calculate sensitive volume

## Pipeline Configuration
The pipeline is configured by two main configuration files. A `.cfg` file is used by `law`, and contains the parameters
for the data generation, export, and inference tasks. See the [`bbh.cfg`](./configs/bbh.cfg) for a complete example.
Training configuration is handled by [`lightning`](https://lightning.ai/docs/pytorch/stable/), which 
uses a `yaml` file. See the [`config.yaml`](../../../projects/train/config.yaml) that lives at the root of the train project for an example. 

> **_Note: Parameters that are common between training and other tasks (e.g. ifos, highpass, fduration) are specified once in the `.cfg` and automatically passed to the downstream training or tuning `config.yaml` by `luigi`/`law`_**

The `aframe-init` command line tool can be used to initialize a directory with configuration files for a fresh run. In the specified directory, `aframe-init` will create default `.cfg` and `.yaml` configuration files, as well as a `run.sh` file for launching the pipeline. Ensure you are in the root of the repository when using `aframe-init`.

```bash
poetry run aframe-init sandbox --directory ~/aframe/my-first-run/ 
```

You can also initialize a directory for launching the tune pipeline

```bash
poetry run aframe-init tune --directory ~/aframe/my-first-tune-run/ 
```

You can now edit these files as you wish.

In the created `run.sh` file, the below environment variables set based on the intialization directory.
They are used to control locations of data and analysis artifcats throughout the run.

- `AFRAME_TRAIN_DATA_DIR` Training data storage
- `AFRAME_TEST_DATA_DIR` Testing data storage
- `AFRAME_TRAIN_RUN_DIR` Training artifact storage
- `AFRAME_CONDOR_DIR` Condor submit files and logs
- `AFRAME_RESULTS_DIR` Inference and sensitive volume results
- `AFRAME_TMPDIR` Intermediate data product storage 

## Running the Pipeline
> **_NOTE: Running the sandbox pipeline out-of-the-box requires access to an enterprise-grade GPU(s) (e.g. P100, V100, T4, A[30,40,100], etc.). There are several nodes on the LIGO Data Grid which meet these requirements_**.

The bottom of the `run.sh` contains the command that launches the pipeline. It should look something like

```bash
LAW_CONFIG_FILE=~/aframe/my-first-run/sandbox.cfg \
poetry run --directory $AFRAME_REPO \
law run aframe.pipelines.sandbox.Sandbox --workers 5 --gpus 0,1
```

Edit the `workers` and `gpus` arguments to suit your needs.

The `workers` argument specifies how many `luigi` workers to use. This controls how many concurrent tasks 
can be launched. It is useful to specify more than 1 worker when you have several tasks that are not dependent on one another. 

The `gpus` argument controls which gpus to use for training and inference. Under the hood, the pipeline is simply setting
the `CUDA_VISIBLE_DEVICES` environment variable. 

The pipeline can now be kicked off by executing the `run.sh` 

```bash
bash ~/aframe/my-first-run/run.sh
```

The end to end pipeline can take a few days to run. The most time consuming steps are training and performing inference. If you wish to reduce these timescales for testing the end-to-end analysis, consider altering the following arguments:
- [`max_epochs`](../../../projects/train/config.yaml#92) in the training `yaml` configuration file
- the amount of analyzed background livetime ([`Tb`](./configs/base.cfg#17)) 
- the number of injections ([`num_injections`](./configs/base.cfg#101))


## Remote Training
If you've successfully followed the [ml4gw quickstart](https://github.com/ml4gw/quickstart/),
model training or tuning can be run on the nautilus hypercluster. 

To initialize your run directory for a remote run, use the optional `--s3-bucket` argument to `aframe-init`

```bash
poetry run aframe-init sandbox --directory ~/aframe/my-first-run --s3-bucket s3://my-bucket/my-first-run
```

This will configure the `AFRAME_TRAIN_RUN_DIR` and `AFRAME_TRAIN_DATA_DIR` in the `run.sh` to point to the specified remote s3 bucket.

The `luigi`/`law` `Tasks` responsible for training data generation will automatically transfer your data to s3 storage, and launch a remote training job
using kubernetes. The rest of the pipeline (export, inference, etc.) is compatible with s3 storage and will work out of the box.


## Tips and Tricks
It is also possible to train locally on the LDG using a remote s3 dataset. The training code will automatically download the dataset from s3.


If you wish to launch an analysis with the freedom of ending
your ssh session, you can use a tool like [`tmux`](https://github.com/tmux/tmux/wiki). Note that `tmux`
is already installed on the LDG clusters.

First, create a new `tmux` session 

```bash
tmux new -s my-first-run
```

Then, launch the pipeline

```bash
bash ~/aframe/my-first-run/run.sh
```

Use `Ctrl` + `b` `d` to exit the `tmux` session. To re-attach simply run

```bash
tmux a
```

to attach to your latest session, or 

```bash
tmux a -t my-first-run
```

to re-attach to a session by name
