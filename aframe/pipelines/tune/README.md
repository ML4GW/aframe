# Hyperparameter Tuning Aframe
> **_Note_** You should have already completed the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) which will help setup access to Nautilus hypercluster resources

`Aframe` utilizes a distributed computing framework [ray](https://docs.ray.io/en/latest/index.html) (more specifically,  the [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) sub-package) for
conducting large scale hyperparameter tuning jobs. Please read through the Ray Tune docs to get familiar with how Ray Tune operates. 

## Tune Pipeline
The `Tune` pipeline `Task` is a wrapper around the [tune command line interface](../../../projects/train/train/tune/cli.py), defined in the `Aframe` train project. It will manage launching, connecting to, and tearing down Ray clusters deployed via kubernetes on the Nautilus hypercluster.

In short, the `Tune` pipeline will do the following:

1. Generate training data 
2. Generate testing data
3. Spin up a Ray cluster using kubernetes
4. Launch a RayTune job on that cluster
5. Export best performing model 
6. Perform inference locally using Triton
7. Calculate sensitive volume

To launch the `Tune` pipeline

```
LAW_CONFIG_FILE=/path/to/config.cfg poetry run law run aframe.pipelines.tune.Tune
```
