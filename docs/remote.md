Remote Training
===============
```{eval-rst}
.. note::
    It is recommended you are familiar with running a :doc:`local pipeline <first_pipeline>` before proceeding
```

```{eval-rst}
.. important::
    For remote training, you must have completed `ml4gw quickstart <https://github.com/ml4gw/quickstart/>`_ instructions, or installed the equivalent software. 
    Specifically, configuring :code:`s3` and `Kubernetes <https://kubernetes.io/>`_ for access to the nautilus hypercluster is required. 
    It is also recommended that you are familiar with Nautilus and `Kubernetes <https://kubernetes.io/>`_. 
    If you are not, the Nautilus introduction `tutorial <https://ucsd-prp.gitlab.io/userdocs/tutorial/introduction/>`_
    is a good place to start.
```

Remote experiments can be initialized using the `aframe-init` command line tool.
To initialize an experiment directory for a remote run, specify the `--s3-bucket` argument `aframe-init`.

```bash
poetry run aframe-init offline --mode sandbox --directory ~/aframe/my-first-remote-run --s3-bucket s3://my-bucket/my-first-remote-run
```

This will configure the `AFRAME_TRAIN_RUN_DIR` and `AFRAME_TRAIN_DATA_DIR` in the `run.sh` to point to the specified remote s3 bucket.

```bash
#!/bin/bash
# Export environment variables
export AFRAME_TRAIN_DATA_DIR=s3://my-bucket/my-first-remote-run/data/train
export AFRAME_TEST_DATA_DIR=/home/albert.einstein/aframe/my-first-remote-run/data/test
export AFRAME_TRAIN_RUN_DIR=s3://my-bucket/my-first-remote-run/training
export AFRAME_CONDOR_DIR=/home/albert.einstein/aframe/my-first-remote-run/condor
export AFRAME_RESULTS_DIR=/home/albert.einstein/aframe/my-first-remote-run/results
export AFRAME_TMPDIR=/home/albert.einstein/aframe/my-first-remote-run/tmp/

# launch pipeline; modify the gpus, workers etc. to suit your needs
# note that if you've made local code changes not in the containers
# you'll need to add the --dev flag!
LAW_CONFIG_FILE=/home/albert.einstein/aframe/my-first-remote-run/sandbox.cfg poetry run --directory /home/albert.einstein/projects/aframev2 law run aframe.pipelines.sandbox.Sandbox --workers 5 --gpus 0
```

The `luigi`/`law` `Tasks` responsible for training data generation will automatically transfer your data to s3 storage, and launch a remote training job using kubernetes. 

```{eval-rst}
.. note:
    Only training is run remotely. The rest of the pipeline (data generation, export, inference, etc.) is run locally. 
    All tasks are able to interact with the s3 artifacts created by the remote training job.
```

## Configuring Remote Resources
The quantity of remote resources can be configured in the `.cfg` config file under the `[luigi_Train]` header

```cfg
[luigi_Train]
...
request_gpus = 4 # number of gpus to request
cpus_per_gpu = 12 # cpus per gpu
memory_per_cpu = 1 # memory in GB
```

It is also possible to sync remote `Aframe` code from git into the container. This is often useful when you are testing an idea that hasn't made 
it onto the `Aframe` `main` branch (and thus hasn't been pushed to the remote container image). To do so, specify the following 
in the `.cfg`.


```cfg
[luigi_Train]
...
# use kubernetes initContainer to sync code
use_init_containers = True
# path to remote git repository
git_url = git@github.com:albert.einstein/aframev2.git
# reference (e.g. branch or commit) to checkout
git_ref = my-feature
```

```{eval-rst}
.. important:
    The git-sync initContainer uses your ssh key to clone software from github. To do so, a Kubernetes secret 
    is made to mount your ssh key into the container. By default, :code:`Aframe` will automatically pull your ssh key from
    :code:`~/.ssh/id_rsa` or :code:`~/.ssh/id_ed25519`
```

## Local Training with S3 Data
Sometimes there are instances where you have data that lives on an `s3` filesystem, but you wish to train using local resources. To do so, 
set `AFRAME_TRAIN_RUN_DIR` to a local path and `AFRAME_TRAIN_DATA_DIR` to an `s3://` location. The training project will detect that the specified data
lives on `s3`, and download it. 


```bash
#!/bin/bash

# remote s3 data 
export AFRAME_TRAIN_DATA_DIR=s3://my-bucket/remote-data-local-training/data/train
# local training 
export AFRAME_TRAIN_RUN_DIR=/home/albert.einstein/remote-data-local-training/training
...
```
