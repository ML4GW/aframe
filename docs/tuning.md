Tuning
======
Hyperparameter tuning is powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). We utilize a wrapper library, [lightray](https://github.com/ethanmarx/lightray), that simplifies the use of Ray Tune with the PyTorch Lightning's `LightningCLI` which is used by `Aframe`.


## Overview
In short, the tuning pipeline will configure a `Ray` "cluster" consisting of worker processes (for executing the training trials) 
and a head process (for scheduling and orchestrating the trials). The beauty of `Ray` is that 

Once the head node and at least one work node are in the "Running" state, the ip address of the head node will be queried, and `lightray` will connect to the `Ray`
cluster and launch the job.

```{eval-rst}
.. note:
    Tuning jobs can be run with local resources, or on the Nautilus hypercluster. Please see the `ml4gw quickstart <https://github.com/ml4gw/quickstart/>`_ to 
    set up the necessary software and credentials to run on Nautilus.
```

## Initialize a Tune Experiment
Setting up a tuning pipeline can be done with the `aframe-init` command line tool. It is very similar to setting up the {doc}`Sandbox </first_pipeline>` pipeline.

```console
poetry run aframe-init offline  --mode tune --directory ~/aframe/my-first-tune/ 
```

Similar to the `Sandbox` pipeline, `.cfg`, `.yaml` and `run.sh` files that configure the tuning experiment will be instantiated in the experiment directory.

## Configuring a Tuning Experiment
The `lightray` library provides an interface to configure most of the parameter exposed by the [`ray.tune.Tuner`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html) class. This includes the scheduler, search algorithm, and other ray tune specific configuration. These can be configured in the `tune.yaml` file.


### Search Parameter Space
A key component of the tuning experiment is the paramter space that is searched over. The parameter space can be set
via the `param_space` attribute in the `tune.yaml` file.

```yaml
# tune.yaml

param_space:
    model.learning_rate: tune.loguniform(1e-4, 1e-1),
    data.kernel_length: tune.choice([1, 2])
```

the parameter names should be python "dot paths" corresponding to attributes in the `train.yaml` file. Any
parameters set in the search space will be sampled from the specified distribution
when each trial is launched, and override the value set in the `train.yaml`.

### Local Tuning
If `AFRAME_TRAIN_RUN_DIR` is set to a local path, then tuning will be performed using local resources. A local ray cluster
will be initialized, and trials will be distributed across available resources.

### Remote Tuning
If `AFRAME_TRAIN_RUN_DIR` is set to an `s3://` path, then tuning will be performed on Nautilus, which uses [Kubernetes](https://kubernetes.io/) for scheduling access to resources. In short, a helm chart is used to launch worker pods and a head pod. Once at the head pod and at least one worker pod are `RUNNING`, the search is started. 

```{eval-rst}
.. note:
    It is recommended you are familiar with Kubernetes and Nautilus. 
    If you are not, the Nautilus introduction `tutorial` <https://ucsd-prp.gitlab.io/userdocs/tutorial/introduction/>`_
    is a good place to start.
```
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

The above configuration will create 4 worker pods, each with 2 gpus for a total of 8 gpus.

```{eval-rst}
..note:
    If for some reason the tuning job fails, you can simply re-launch the pipeline and `Ray` will automatically restart the trials
    from the latest experiment state
```

### Syncing Remote Code
In some cases, it is necessary to launch a tuning job with code changes that haven’t been integrated into the `Aframe` `main` branch, and thus have not been pushed to the remote container. To allow this, the `lightray/ray-cluster` `helm` chart supports an optional [git-sync](https://github.com/kubernetes/git-sync) `initContainer` that will clone and mount remote code from github inside the kubernetes pods. The remote repository and reference can be configured under the `[luigi_TuneRemote]` header in the `tune.cfg`

```cfg
[luigi_TuneRemote]
...
# path to remote git repository
git_url = git@github.com:albert.einstein/aframe.git
# reference (e.g. branch or commit) to checkout
git_ref = my-feature
```

```{eval-rst}
.. important::
    The git-sync :code:`initContainer` uses your ssh key to clone software from github. To do so, a Kubernetes secret 
    is made to mount your ssh key into the container. By default, :code:`Aframe` will automatically pull your ssh key from
    :code:`~/.ssh/id_rsa`. You can override this default under the :code:`[luigi_ssh]` header

    .. code-block:: ini
    
        [luigi_ssh]
        ssh_file = ~/.ssh/id_ed25519
```

## Restoring an Experiment
