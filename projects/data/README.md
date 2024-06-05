# Data
Scripts for producing training and testing data for `Aframe`

## Environment
The `data` project environment utilizes `Mamba` and `poetry`. `Mamba` is needed for installing
the LIGO frame reading libraries [`python-ldas-tools-framecpp`](https://anaconda.org/conda-forge/python-ldas-tools-framecpp/) and [https://anaconda.org/conda-forge/python-nds2-client](`python-nds2-client`), which are unavailable on PyPi.

In the root of the `data` project, run 
```bash
apptainer build $AFRAME_CONTAINER_ROOT/data.sif apptainer.def
```
to build the `data` container. 

The container will first build an environment using the [`conda-lock.yml`](./conda-lock.yml), and then install local dependencies defined in the [`pyproject.toml`](./pyproject.toml).

If the dependencies in the [`environment.yaml`](./environment.yaml) require modifications, the `conda-lock.yml` will need to be updated

```bash
conda-lock -f environment.yaml -p linux-64
```

and the container image will need to be rebuilt.

## Scripts
The data project consists of four main sub-modules:

1. `data/segments` - Querying science mode segments
2. `data/fetch` - Fetching strain data
3. `data/timeslide_waveforms` - Generating waveforms for injection campaigns
4. `data/waveforms` - Generating waveforms for training Aframe

Additionally, the main executable of each sub-module is exposed via a CLI at `data/cli.py`

## Example: generating training data
As an example, let's build a training dataset using the CLI in the `data` container we built above

First, let's make a data storage directory, and query science mode segments from [gwosc](gwosc.org)
```bash
mkdir -p ~/aframe/data/train
apptainer run $AFRAME_CONTAINER_ROOT/data.sif \
    python -m data query --flags='["H1_DATA", "L1_DATA"]' --start 1240579783 --end 1241443783 --output_file ~/aframe/data/segments.txt
```

Inspecting the output, (`vi ~/aframe/data/segments.txt`) it looks like there are science mode data segments between `(1240579783, 1240587612)` and `(1240594562, 1240606748)`. 

Next, let's fetch strain data during those segments. One will be used for training, the other for validating

```bash
apptainer run $AFRAME_CONTAINER_ROOT/data.sif \
    python -m data fetch \
    --start 1240579783 \
    --end 1240587612 \
    --channels='["H1", "L1"]' \
    --sample_rate 2048 \
    --output_directory ~/aframe/data/train/background/

apptainer run $AFRAME_CONTAINER_ROOT/data.sif \
    python -m data fetch \
    --start 1240594562 \
    --end 1240606748 \
    --channels='["H1", "L1"]' \
    --sample_rate 2048 \
    --output_directory ~/aframe/data/train/background/
```

Finally, lets generate some waveforms for training

```bash
apptainer run $AFRAME_CONTAINER_ROOT/data.sif \
    python -m data waveforms \
    --prior priors.priors.end_o3_ratesandpops \
    --num_signals 10000 \
    --waveform_duration 8 \
    --sample_rate 2048 \
    --output_file ~/aframe/data/train/train_waveforms.hdf5
```

and validation

```bash
apptainer run $AFRAME_CONTAINER_ROOT/data.sif \
    python -m data waveforms \
    --prior priors.priors.end_o3_ratesandpops \
    --num_signals 2000 \
    --waveform_duration 8 \
    --sample_rate 2048 \
    --output_file ~/aframe/data/train/val_waveforms.hdf5
```

Note that the train project assumes these waveform files are named as above! To continue this example, see the [training `Aframe` example](../train/README.md#example-training-aframe)
