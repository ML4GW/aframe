# Sandbox Pipeline

> **_Note: It is highly recommended that you have completed the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) instructions before running the sandbox pipeline_**

> **_Note: It is assumed that you have already built each [project's container](../../../projects/README.md)_**

The sandbox pipeline strings together `luigi` / `law` tasks to run an end-to-end generic aframe workflow.
In short, running the sandbox pipeline will

1. Generate training data
2. Generate testing data
3. Train a model
4. Export trained weights to TensorRT
5. Perform inference using Triton
6. Calculate sensitive volume

## Pipeline Configuration
The pipeline is configured by two files. A `.cfg` file contains the parameters
for the data generation, export, and inference tasks. See the [`sandbox.cfg`](./sandbox.cfg) for a complete example.
Training configuration is handled by [`lightning`](https://lightning.ai/docs/pytorch/stable/), which 
uses a `yaml` file. By default, the Sandbox pipeline will use the [`config.yaml`](../../../projects/train/config.yaml) that lives at the root of the train project. If you wish to point to a different location, you can set the `config` parameter of the `luigi_Train` task in the `.cfg`:

```cfg
[luigi_Train]
config = /path/to/training/config.yaml
```

> **_Note: Parameters that are common between training and other tasks (e.g. ifos, highpass, fduration) are specified once in the `.cfg` and passed to the downstream training `config.yaml` by `luigi`/`law`_**

Lastly, environment variables are used to control locations of data and analysis artifcats throughout the run:

- `AFRAME_TRAIN_DATA_DIR`: Training data storage
- `AFRAME_TEST_DATA_DIR`: Testing data storage
- `AFRAME_TRAIN_RUN_DIR`: Training artifact storage
- `AFRAME_CONDOR_DIR`: Condor submit files and logs
- `AFRAME_RESULTS_DIR`: Inference and sensitive volume results

It is recommended to store these in a `.env` file. The following pattern could prove useful:

```bash
AFRAME_BASE=~/aframe # base location for aframe runs
RUN_LABEL=my_first_run # some descriptive label for this analysis
export AFRAME_TRAIN_DATA_DIR=$AFRAME_BASE/$RUN_LABEL/data/train
export AFRAME_TEST_DATA_DIR=$AFRAME_BASE/$RUN_LABEL/data/test
export AFRAME_TRAIN_RUN_DIR=$AFRAME_BASE/$RUN_LABEL/training
export AFRAME_CONDOR_DIR=$AFRAME_BASE/$RUN_LABEL/condor
export AFRAME_RESULTS_DIR=$AFRAME_BASE/$RUN_LABE/results
```

To export these environment variables, simply run

```
source .env
```

## Running the Pipeline
To launch the pipeline, ensure that you are in the root level of the repository.
Then, the pipeline can be launched using `poetry` and the `law` command line tool.

```
LAW_CONFIG_FILE=/path/to/sandbox.cfg \
    poetry run law run aframe.pipelines.sandbox.Sandbox
    --gpus 0,1 --workers 5
```

The `workers` argument specifies how many `luigi` workers to use. This controls how many concurrent tasks 
can be launched. It is useful to specify more than 1 worker when you have tasks that are not dependent on one another. 

The `gpus` argument controls which gpus to use for training and inference. Under the hood, the pipeline is simply setting
the `CUDA_VISIBLE_DEVICES` environmentt variable. 

The end to end pipeline can take a few days to run. If you wish to launch an analysis with the freedom of ending
your ssh session, it is recommended that you use a tool like [`tmux`](https://github.com/tmux/tmux/wiki). Note that `tmux`
is already installed on the LDG clusters.

First, create a new session and change directories into the aframe repository
```
tmux new -s aframe-sandbox
```

Next, ensure that you've re-sourced your .env!
```
source /path/to/.env
```

Finally, launch the analysis
```
LAW_CONFIG_FILE=/path/to/sandbox.cfg \
    poetry run law run aframe.pipelines.sandbox.Sandbox
    --gpus 0,1 --workers 5
```

Use `Ctrl` + `b` `d` to exit the `tmux` session. To re-attach simply run

```
tmux a
```
to attach to your latest session, or 

```
tmux a -t aframe-analysis
```

to re-attach to a session by name




    
