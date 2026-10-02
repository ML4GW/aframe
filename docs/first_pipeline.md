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
5. Perform inference on timeshifted background and injections
6. Calculate sensitive volume

## Environment setup
The pipeline is orchestrated by Snakemake, which runs in its own `conda` environment. With [conda-lock](https://github.com/conda/conda-lock) installed, in the root of this repository run

```console
conda-lock install --name aframe-smk pipeline/envs/snakemake.conda-lock.yml
conda activate aframe-smk
```

Snakemake also needs a handful of environment variables, which are best set in your shell config (e.g. `~/.bashrc`):

```bash
export AFRAME_CONTAINER_ROOT=~/aframe/images   # where the images are built
export LIGO_USERNAME=albert.einstein           # LDG accounting
export LIGO_GROUP=ligo.dev.o4.cbc.allsky.aframe
export WANDB_API_KEY=...                       # if the train config logs to W&B
```

Under the `ldg-transfer` profile, condor jobs run in images published to OSDF. After building the images, publish them from a CIT access point with

```console
python -m scripts.publish_images
```

## Configuration
The pipeline is configured by two files. A `config.yaml` is used by Snakemake, and contains the parameters for data generation, export, inference and plotting. See [here](https://github.com/ML4GW/aframe/blob/main/pipeline/config/config.yaml) for a complete example. `pipeline/config/config.yaml` holds the defaults, so a run's `config.yaml` only lists what it changes.

Training is configured by [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), which uses its own `.yaml` file. See [here](https://github.com/ML4GW/aframe/blob/main/projects/train/train.yaml) for a complete example.

```{eval-rst}
.. note::
    Parameters common to training and other rules (e.g. :code:`ifos`, :code:`highpass`, :code:`fduration`) are set once in the pipeline :code:`config.yaml` and passed to each rule by Snakemake.
```

## Initialize a Pipeline
The `aframe-init` command line tool creates a run directory with a `config.yaml`, a `train.yaml`, and a `run.sh` that launches the pipeline. Everything the run writes goes in this directory:

- `data/` Background strain data
- `waveforms/` Training, validation, and testing waveforms
- `train/`, `export/`, `triton/`, `infer/`, `plots/` Experiment outputs
- `logs/` Rule logs
- `code/` The read-only copy of the local repo that the run uses

```{eval-rst}
.. tip::
    When running a new "experiment", it is recommended to use :code:`aframe-init` to initialize a new directory. This way, all the configuration associated with the experiment is isolated, and the experiment is reproducible.
```

While in the root directory, a pipeline can be initialized with

```console
uv run aframe-init snakemake -d ~/aframe/my-first-run/
```

```{eval-rst}
.. tip::
    Add :code:`--preset small` for a minimal run that can be used to check that the pipeline works before a full run.
```

By default the run uses the LIGO Data Grid profile, for execute points that mount `/home`. Pass `-p`/`--profile` to pick another: `pipeline/profiles/ldg-transfer` for execute points that share nothing with the submit node, or `pipeline/profiles/delta` for Delta.

Now, you can navigate to the experiment directory and edit the configuration files as you wish.

## Running the Pipeline
```{eval-rst}
.. note::
    Training, export, and Triton inference run on the node where the pipeline is launched, which therefore needs enterprise-grade GPUs (e.g. V100, T4, A[10,30,40,100]). There are several such nodes on the LIGO Data Grid.
```

Launch the pipeline from the run directory with

```console
cd ~/aframe/my-first-run
./run.sh
```

`run.sh` copies the files git knows about from your local repo into the run's read-only `code/` directory, then runs Snakemake from that copy. Edits to the local repo therefore take effect the next time the run starts. Arguments to `run.sh` are passed to Snakemake, so `./run.sh -n` does a dry run, printing the rules that would execute without running them. To resume an interrupted run, run `./run.sh` again.

Which GPUs training, export and the Triton server use is set by `train_gpus` and `inference_gpus` in `config.yaml`. These are either a comma-separated list of IDs (e.g. `"0,1"`), or `auto` for the least-used GPUs when each rule starts.

```{eval-rst}
.. tip::
    The end to end pipeline can take a few days to run.
    If you wish to launch an analysis with the freedom of ending
    your ssh session, use a tool like `tmux <https://github.com/tmux/tmux/wiki>`_ or `screen <https://www.gnu.org/software/screen/manual/screen.html>`_
```

The most time consuming steps are training and inference. To shorten them, consider reducing:
- `max_epochs` and `batches_per_epoch` in `train.yaml`
- the background livetime `Tb` and the number of injections `num_testing_signals` in `config.yaml`

See [pipeline/README.md](https://github.com/ML4GW/aframe/blob/main/pipeline/README.md) for where rules run and the containers.
