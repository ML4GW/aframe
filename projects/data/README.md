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
```
mkdir ~/aframe/data/
apptainer run ~/aframe/images/data.sif \
    python -m data query --flags='["H1_DATA", "L1_DATA"]' --start 1240579783 --end 1241443783 --output_file ~/aframe/data/segments.txt
```

Inspecting the output, (`vi ~/aframe/data/segments.txt`) it looks like there are science mode data segments between `(1240579783, 1240587612)` and `(1240594562, 1240606748)`. 

Next, let's fetch strain data during those segments. One will be used for training, the other for validating

```
apptainer run ~/aframe/images/data.sif \
    python -m data fetch \
    --start 1240579783 \
    --end 1240587612 \
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

Finally, lets generate some waveforms for training

```
apptainer run ~/aframe/images/data.sif \
    python -m data waveforms \
    --prior priors.priors.end_o3_ratesandpops \
    --num_signals 10000 \
    --waveform_duration 8 \
    --sample_rate 2048 \
    --output_file ~/aframe/data/signals.hdf5
```
