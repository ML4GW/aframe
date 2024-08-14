# Online Deployment
Scripts for deploying `Aframe` and `AMPLFI` models for online analysis  

## Getting Started

> **Note** In the below setup, we will be using the CLI tools installed in the root `Aframe` environment

First build the online deployment apptainer image
```
poetry run build-containers online
```

Next, you can initialize an online run directory, which will add template `config.yaml` and `run.sh` files that will need to be populated with arguments specific to your analysis.

```
poetry run aframe-init online --directory /path/to/online
```

Once the files are populated correctly, the analysis can be launched via

```console
bash run.sh
```

## Uploading Events to GraceDB

Uploading events to gracedb requires the use of [scitokens](https://computing.docs.ligo.org/guide/auth/scitokens/?h=sci#scitokens)

If you do not have access to GraceDB but still want to launch an analysis, you can set

```
server = "local"
```

in the configuration, which will write events to disk instead of uploading to GraceDB

## Crontab

A `crontab` file is also added to the directory, which will automatically refresh scitokens used for gracedb authentication, as well as relaunch `run.sh` if there are any failures. Make sure to adjust the token refresh commands with your own credentials, and then you can launch the analysis with

```
crontab crontab
```
