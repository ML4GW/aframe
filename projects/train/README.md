# Train
Training Aframe networks using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and hyper-parameter tuning using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) 

## Environment
The train project environment is manged entirely by `poetry`.

In the root of the `train` project, run 
```bash
apptainer build $AFRAME_CONTAINER_ROOT/train.sif apptainer.def
```
to build the `train` container.

This project can also be installed locally via 

```
poetry install
```

## Scripts
The train project consists of two main executables

`train` - launch a single training job
`train.tune` - launch distributed hyper-parameter tuning using Ray

### Train
The training script takes advantage of [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli) allowing for  modularity and flexibility. One single training script supports

- Time domain and Frequency domain data representations
- Supervised and Semi-supervised optimization schemes

all by changing a configuration file. This is achieved by using a class hierarchy of [`DataModules`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) and [`LightningModules`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) where core functionality 
common to all use-cases is abstracted into base classes. 

To see a list of arguments one can locally run 

```bash
poetry run python -m train --help
```

or inside the container

```bash
apptainer run $AFRAME_CONTAINER_ROOT/train.sif python -m train --help
```

This list can be overwhelming, and it is suggested that you start from the default [configuration file](./config.yaml).


#### Example: Training Aframe

> **Note** It is assumed you have generated a training dataset via the [data project example](../data/README.md#example-generating-training-data)

The following will a training run using GPU 0

```bash
mkdir ~/aframe/results
APPTAINERENV_CUDA_VISIBLE_DEVICES=0 apptainer run --nv $AFRAME_CONTAINER_ROOT/train.sif \
    python -m train \
        --config /opt/aframe/projects/train/config.yaml \
        --data.ifos=[H1,L1] \
        --data.data_dir ~/aframe/data/train \
        --trainer.logger=WandbLogger \
        --trainer.logger.project=aframe \
        --trainer.logger.name=my-first-run \
        --trainer.logger.save_dir=~/aframe/results/my-first-run
```

This will infer most of your training arguments from the YAML config that got put into the container at build time. If you want to change this config, or if you change any code and you want to see those changes reflected inside the container, you can "bind" your local version of the [root](../../) `Aframe` repository into the container by including `apptainer run --bind .:/opt/aframe` at the beginning of the above command. 

Once your run is started, you can go to [wandb.ai](https://wandb.ai) and track your loss and validation score. If you don't want to track your run with W&B, just remove all the first three `--trainer` arguments above. This will save your training metrics to a local CSV in the `save_dir`.

You can even train using multiple GPUS for free! Just specify a list of comma-separated GPU indices to `APPTAINERENV_CUDA_VISIBLE_DEVICES`.

### Tune
In addition, the train project consists of a tuning script for performing a distributed hyper-parameter search with Ray Tune. 
It is recommended that multiple GPU's are available for an efficient search.

A local tune job can be launched with 
```
APPTAINERENV_CUDA_VISIBLE_DEVICES=<IDs of GPUs to tune on> apptainer run --nv $AFRAME_CONTAINER_ROOT/train.sif \
    python -m train.tune \
        --config /opt/aframe/projects/train/config.yaml
        --data.ifos=[H1,L1]
        --data.data_dir ~/aframe/data/train
        --trainer.logger=WandbLogger
        --trainer.logger.project=aframe
        --trainer.logger.save_dir=~/aframe/results/my-first-tune \
        --tune.name my-first-tune \
        --tune.storage_dir ~/aframe/results/my-first-tune \
        --tune.temp_dir ~/aframe/results/my-first-tune/ray \
        --tune.num_samples 8 \
        --tune.cpus_per_gpu 6 \
        --tune.gpus_per_worker 1 \
        --tune.num_workers 4
```
This will launch 8 hyperparameter search jobs that will execute on 4 workers using the Asynchronous Successive Halving Algorithm (ASHA).
All the runs will be given the same **group** ID in W&B, and will be assigned random names in that group.

**NOTE: for some reason, right now this will launch one job at a time that takes all available GPUs. This needs sorting out**

If you already have a ray cluster running somewhere, you can distribute your jobs over that cluster by simply adding the `--tune.endpoint <ip address of ray cluster>:10001` command line argument.





Similarly, to see a list of arguments one can locally run 

```bash
poetry run python -m train.tune --help
```

or inside the container

```bash
apptainer run $AFRAME_CONTAINER_ROOT/train.sif python -m train.tune --help
```
