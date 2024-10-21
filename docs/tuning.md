Tuning
======
Hyperparameter tuning is powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). We utilize a wrapper library, [lightray](https://github.com/ethanmarx/lightray), that simplifies the use of Ray Tune with the PyTorch Lightning's `LightningCLI` which is used by `Aframe`.

```{eval-rst}
.. note:
    Currently, tuning jobs are only able to be run on the Nautilus hypercluster. Please see the `ml4gw quickstart <https://github.com/ml4gw/quickstart/>`_ to 
    set up the necessary software and credentials to run on Nautilus.
```

## Overview
The Nautilus hypercluster uses [Kubernetes](https://kubernetes.io/) for scheduling access to resources. 

```{eval-rst}
.. note:
    It is recommended you are familiar with Kubernetes and Nautilus. 
    If you are not, the Nautilus introduction `tutorial` <https://ucsd-prp.gitlab.io/userdocs/tutorial/introduction/>`_
    is a good place to start.
```

In short, the tuning pipeline will configure a `Ray` "cluster" on Nautilus consisting of worker pods (for performing executing the training jobs) 
and a head pod (for scheduling and orchestrating the trials).

Once the head node and at least one work node are in the "Running" state, the ip address of the head node will be queried, and `lightray` will connect to the `Ray`
cluster and launch the job.


## Initialize a Tune Experiment
Setting up a tuning pipeline can be done with the `aframe-init` command line tool. It is very similar to setting up the {doc}`Sandbox </first_pipeline>` pipeline.

```console
poetry run aframe-init offline  --mode tune --directory ~/aframe/my-first-tune/ 
```

Like the `Sandbox` pipeline, a `.cfg`, `.yaml` and `run.sh` will be instantiated in the directory where you
created the experiment. In addition, a `search_space.py` file will be created. More on this below.


## Configuring a Tuning Experiment

### Search Space
The search space of parameters to tune over can be set in the `search_space.py` file. 
For example, the below parameter space will search over the models learning rate 
and the kernel length of the data.

```
# search_space.py
from ray import tune

space = {
    "model.learning_rate": tune.loguniform(1e-4, 1e-1),
    "data.kernel_length": tune.choice([1, 2])
}
```

the parameter names should be python "dot path" to attributes in the training `.yaml`. Any
parameters set in the search space will be sampled from the specified distribution
when each trial is launched, and override the value set in the `.yaml`.

### Remote Resources
The number of worker pods, gpus per pod, and other resource configuration can be specified under the 
`[luigi_ray_worker]` and `[luigi_ray_head]` headers in the `tune.cfg` file. See the [ray worker](https://github.com/ML4GW/aframe/blob/main/aframe/config.py#L16)
and [ray head](https://github.com/ML4GW/aframe/blob/main/aframe/config.py#L48) `luigi.Config` objects for all possible configuration.


```cfg
[luigi_ray_worker]
# request 4 worker pods
replicas = 4 
# number of gpus per replica
gpus_per_replica = 2


[luigi_ray_head]
cpus = 32
memory = 32G
```

this configuration will create 4 worker pods, each with 2 gpus for a total of 8 gpus.∂ß

### Syncing Remote Code
In some cases, it is necessary to launch a tuning job with code changes that haven’t been integrated into the `Aframe` `main` branch, and thus have not been pushed to the remote container. To allow this, the `lightray/ray-cluster` `helm` chart supports an optional git-sync initContainer that will clone and mount remote code from github inside the kubernetes pods.
The remote repository and reference can be configured under the `[luigi_TuneRemote]` header in the `tune.cfg`

```cfg
[luigi_TuneRemote]
...
# path to remote git repository
git_url = git@github.com:albert.einstein/aframe.git
# reference (e.g. branch or commit) to checkout
git_ref = my-feature
```


```{eval-rst}
.. important:
    The git-sync initContainer uses your ssh key to clone software from github. To do so, a Kubernetes secret 
    is made to mount your ssh key into the container. By default, :code:`Aframe` will automatically pull your ssh key from
    :code:`~/.ssh/id_rsa` or :code:`~/.ssh/id_ed25519`.
```
