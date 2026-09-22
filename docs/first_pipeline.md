First Pipeline
==============

```{eval-rst}
.. note::
    It is highly recommended that you have completed the `ml4gw quickstart <https://github.com/ml4gw/quickstart/>`_ instructions, or installed the equivalent software, before running the pipeline.
```

```{eval-rst}
.. note::
    It is assumed that you have already built each project's container (See :doc:`projects <projects>`)
```

Aframe pipelines string together [Snakemake](https://snakemake.readthedocs.io/) rules to run an end-to-end workflow. Here, we will run the offline pipeline.
In short, the pipeline will

1. Generate training data
2. Generate testing data
3. Train a model
4. Export trained weights to TensorRT
5. Perform inference using Triton
6. Calculate sensitive volume

## Environment setup
The pipeline is orchestrated by Snakemake, which runs in its own `conda` environment. In the root of this repository, run

```console
conda env create -f pipeline/envs/snakemake.yaml
conda activate aframe-smk
```

Snakemake also needs a handful of environment variables: where your container images live, which directories to bind into them, and your LDG account info. Copy the example file and fill it in:

```console
cp pipeline/.env.example pipeline/.env
```

`pipeline/.env` is git-ignored, and is sourced automatically by the `run.sh` described below.

## Configuration
The pipeline is configured by two main configuration files. A `.yaml` file is used by Snakemake, and contains the parameters
for the data generation, export, and inference rules. See [here](https://github.com/ML4GW/aframe/blob/main/pipeline/config/config.yaml) for a complete example.

Training configuration and parsing is handled by [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), which
uses a `.yaml` file. See [here](https://github.com/ML4GW/aframe/blob/main/projects/train/train.yaml) for a complete example

```{eval-rst}
.. note::
    When running pipelines, parameters that are common between the training rule and other rules (e.g. :code:`ifos`, :code:`highpass`, :code:`fduration`) are specified once in the pipeline :code:`config.yaml` and automatically passed to the downstream rules by Snakemake.
```

`pipeline/config/config.yaml` is always loaded first, so a run config only needs to list the parameters you want to change.

## Initialize a Pipeline
The `aframe-init` command line tool can be used to initialize a directory with configuration files for a fresh run.
In the specified directory, `aframe-init` will create default `config.yaml` and `train.yaml` configuration files, as well as a `run.sh` file for launching the pipeline.

```{eval-rst}
.. tip::
    When running a new "experiment", it is recommended to use :code:`aframe-init` to initialize a new directory. This way, all the configuration associated with the experiment is isolated, and the experiment is reproducible.
```

While in the root directory, a pipeline can be initialized with

```console
uv run aframe-init snakemake -d ~/aframe/my-first-run/
```

By default this targets the LIGO Data Grid profile. Pass `-p`/`--profile` to pick a different one, e.g. `--profile pipeline/profiles/local` to run everything as subprocesses on the current machine.

Now, you can navigate to the experiment directory and edit the configuration files as you wish.

## Running the Pipeline
```{eval-rst}
.. note::
    Running the pipeline out-of-the-box requires access to an enterprise-grade GPU(s) (e.g. P100, V100, T4, A[30,40,100], etc.). There are several nodes on the LIGO Data Grid which meet these requirements.
```

In the experiment directory a `run.sh` file will be created that looks like

```bash
#!/bin/bash
# Set AFRAME_DEV=1 to bind the working tree into containers.
cd /home/albert.einstein/projects/aframe
source pipeline/.env
snakemake --configfile /home/albert.einstein/aframe/my-first-run/config.yaml --profile pipeline/profiles/ldg
```

The generated `config.yaml` sets `run_dir` to the experiment directory, and `train_config` to the `train.yaml` beside it. Where pipeline artifacts are stored is controlled by these configuration parameters rather than environment variables:

- `run_dir` Experiment outputs (`train/`, `export/`, `infer/`, `plots/`)
- `background_dir` Background strain data; defaults to `{run_dir}/data`
- `waveforms_dir` Training, validation, and testing waveforms; defaults to `{run_dir}/waveforms`
- `log_dir` Rule logs; defaults to `{run_dir}/logs`

`background_dir` and `waveforms_dir` can be pointed at shared locations so that runs covering the same GPS range and waveform parameters do not regenerate the same data.

The last line of the `run.sh` contains the command that launches the pipeline. The `--profile` argument selects where rules execute — `pipeline/profiles/ldg` submits to HTCondor, `pipeline/profiles/delta` to Slurm, and `pipeline/profiles/local` runs everything as subprocesses. The profile also sets how many jobs may run concurrently, so there is no `workers` argument to tune here.

Which GPUs are used for export and inference is set by the `gpus` parameter in the pipeline `config.yaml`, specified as a comma separated list (e.g. `gpus: "0,1,2"`). Training GPUs are set separately via `trainer.devices` in the training `yaml`.

The pipeline can now be kicked off by executing the `run.sh`

```console
bash ~/aframe/my-first-run/run.sh
```

```{eval-rst}
.. tip::
    It is worth adding :code:`-n` to the :code:`snakemake` command in :code:`run.sh` for a dry run first. This prints the rules that would execute without running any of them.
```

```{eval-rst}
.. tip::
    The end to end pipeline can take a few days to run.
    If you wish to launch an analysis with the freedom of ending
    your ssh session, use a tool like `tmux <https://github.com/tmux/tmux/wiki>`_ or `screen <https://www.gnu.org/software/screen/manual/screen.html>`_
```

The most time consuming steps are training, and performing inference. If you wish to reduce these timescales for testing the end-to-end analysis, consider altering the following arguments:
- Number of training epochs, `max_epochs`, in the training `yaml` configuration file
- Batches analyzed each epoch, `batches_per_epoch`, in the training `yaml` configuration file
- Seconds of analyzed background livetime `Tb`, in the pipeline `config.yaml`
- Number of injections performed, `num_testing_signals`, in the pipeline `config.yaml`
