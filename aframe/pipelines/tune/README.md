# Hyperparameter Tuning Aframe
> **_Note_** You should have already completed the [ml4gw quickstart](https://github.com/ml4gw/quickstart/), providing access to Nautilus hypercluster resources

`Aframe` utilizes a distributed computing framework [ray](https://docs.ray.io/en/latest/index.html) (more specifically,  the ray sub-package [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)) for
conducting large scale hyperparameter tuning jobs. Please read through the Ray Tune docs to get familiar with how Ray Tune operates. 

## Remote Tuning
In short, the `Tune` pipeline will do the following:

1. Generate training data 
2. Generate testing data
3. Spin up a Ray cluster using kubernetes
4. Launch a RayTune job on that cluster
5. Export best performing model 
6. Perform inference locally using Triton
7. Calculate sensitive volume

This `Task` is a wrapper around the [tune command line interface](../../../projects/train/train/tune/cli.py), defined in the train project, that will manage launching, connecting to, and tearing down kubernetes resources on nautilus for the tune job.

Running tune jobs requires access to training data, and thus the `TuneRemote` `Task` `requires()` the `FetchTrain`
and `TrainWaveforms` `Tasks` that will generate strain and simulated waveforms respectively. The easiest way to 
configure all of these tasks is with a configuration file. An [example](./tune.cfg) is provided in this folder.

The `output()` of the `TuneRemote`

To launch the `Tune` pipeline

```
LAW_CONFIG_FILE=/path/to/config.cfg poetry run law run aframe.pipelines.tune.Tune
```
