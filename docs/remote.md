Remote Training
===============

```{eval-rst}
.. important::
    For remote training, you must have completed [ml4gw quickstart](https://github.com/ml4gw/quickstart/) instructions, or installed the equivalent software, before running the sandbox pipeline. Specifically, configuring `s3` and kubernetes for access to the nautilus hypercluster. 
```

Remote experiments can be initialized using the `aframe-init` command line tool.
To initialize an experiment directory for a remote run, specify the `--s3-bucket` argument `aframe-init`.

```bash
poetry run aframe-init offline --mode sandbox --directory ~/aframe/my-first-run --s3-bucket s3://my-bucket/my-first-run
```

This will configure the `AFRAME_TRAIN_RUN_DIR` and `AFRAME_TRAIN_DATA_DIR` in the `run.sh` to point to the specified remote s3 bucket.

The `luigi`/`law` `Tasks` responsible for training data generation will automatically transfer your data to s3 storage, and launch a remote training job using kubernetes. The rest of the pipeline (export, inference, etc.) is run locally. These tasks are able to interact with s3 storage and will work out of the box.


## Local Training with S3 Data
Sometimes there are instances where you have data that lives on an s3 filesystem, but you wish to train locally. To do so,
set `AFRAME_TRAIN_RUN_DIR` to a local path and `AFRAME_TRAIN_DATA_DIR` to an `s3://` location. The training code will detect that the specified data
lives on `s3`, and download it. 
