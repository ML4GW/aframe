# Sandbox Pipeline

**Note: It is highly recommended that you have completed the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) instructions before running the sandbox pipeline**

The sandbox pipeline strings together `luigi` / `law` tasks to run an end-to-end generic aframe workflow.
In short, running the sandbox pipeline will

1. Generate training data
2. Generate testing data
3. Train a model
4. Export trained weights to TensorRT
5. Perform inference using Triton
6. Calculating sensitive volume

## Pipeline Configuration
Most of the pipeline task parameters are configured with a `.cfg` file.
See the [`sandbox.cfg`](./sandbox.cfg) for a complete example.

The only configuration not controlled by the `.cfg` file are locations for file storage. 
Instead, these paths are controlled by the following environment variables:

`AFRAME_TRAIN_DATA_DIR`: Training data storage 
`AFRAME_TEST_DATA_DIR`: Testing data storage
`AFRAME_TRAIN_RUN_DIR`: Training artifact storage
`AFRAME_CONDOR_DIR`: Condor submit files and logs
`AFRAME_RESULTS_DIR`: Inference and sensitive volume results

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
    poetry run law run aframe.pipelines.sandbox.Sandbox'
    --gpus 0,1 --workers 5
```

The end to end pipeline can take a few days to run. If you wish to launch an analysis with the freedom of ending
your ssh session, it is recommended that you use a tool like [`tmux`](https://github.com/tmux/tmux/wiki). Note that `tmux`
is already installed on the LDG clusters.

First, create a new session and change directories into the aframe repository
```
tmux new -s aframe-sandbox
cd /path/to/aframe
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




    
