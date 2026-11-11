Online Deployment
=================

Scripts for deploying `Aframe` and `AMPLFI` models for online analysis  

## Getting Started

> **Note** In the below setup, we will be using the CLI tools installed in the root `Aframe` environment

First build the online deployment apptainer image
```
uv run build-containers online
```

Next, you can initialize an online run directory, which will add template `config.yaml` and `run.sh` files that will need to be populated with arguments specific to your analysis.

```
uv run aframe-init online --directory /path/to/online
```

Once the files are populated correctly, the analysis can be launched via

```console
bash run.sh
```

`run.sh` writes its PID to `run.pid` in the run directory. To stop the search and the monitor, run

```console
kill $(cat run.pid)
```

from the run directory.

## Uploading Events to GraceDB

Uploading events to gracedb requires the use of [scitokens](https://computing.docs.ligo.org/guide/auth/scitokens/?h=sci#scitokens)

If you do not have access to GraceDB but still want to launch an analysis, you can set

```
server = "local"
```

in the configuration, which will write events to disk instead of uploading to GraceDB

## Crontab

A `crontab` file is also added to the directory. It holds a single `@reboot` line that relaunches `run.sh` when the node reboots. Add the line to the account's crontab with

```
crontab -e
```
