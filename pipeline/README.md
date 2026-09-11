# The Aframe Snakemake Pipeline

## Quick start

```bash
# One-time: create the orchestrator environment
conda env create -f pipeline/envs/snakemake.yaml # or from the conda-lock file
conda activate aframe-smk

# One-time: fill in environment variables (container root, data dirs, ...)
cp pipeline/.env.example pipeline/.env   # then edit pipeline/.env (git-ignored)

# One-time: build the per-project containers
uv run build-containers # may need to build one at a time

# Initialize a run directory with a config.yaml, train.yaml, and run.sh
uv run aframe-init snakemake -d /path/to/my-run
# Edit /path/to/my-run/config.yaml to override paramerers in pipeline/config/config.yaml
# Edit /path/to/my-run/train.yaml to adjust model/training hyperparameters.

# Edit run.sh to add flags like -n (dry-run) or --until <rule> before launching.
bash /path/to/my-run/run.sh
```

## Layout

```
Snakefile                     entry point: config + includes + rule all
pipeline/
  config/config.yaml          all pipeline parameters (loaded by default)
  config/review.yaml          short end-to-end validation preset
  profiles/local/             dev/testing: everything runs as subprocesses
  profiles/condor/            LDG HTCondor execution
  envs/snakemake.yaml         orchestrator env
  envs/dev.yaml               local dev env
  .env.example                required environment variables
projects/
  data/data.smk               segment querying, strain fetching, waveform generation
  train/train.smk             local or remote (Nautilus) training (WIP)
  export/export.smk           Export to a Triton model repository
  infer/infer.smk             Triton server + timeslide inference
  plots/plots.smk             sensitive volume
```

Each `.smk` file sits in the project whose CLI it invokes.
The top-level `Snakefile` only stitches them together (note: shared namespace).
Rules never import project code directly, instead using the `shell`  directive
to call the CLI entry points.

Data generation, inference clients, and aggregation scripts run via condor.
Training, export, and the Triton server run on the node where Snakemake is
launched, and so this node must have GPUs.
The branchmap checkpoints and the `stop_triton` rule run within the Snakemake
process itself.

## Checkpoints

Most of the DAG cannot be determined up front: the number of
strain-fetching jobs depends on what segments DQSegDB returns, and the
number of waveform/inference branches depends on which segments are
long enough to analyze. Snakemake's equivalent is
the [checkpoint](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#data-dependent-conditional-execution)
mechanism, and the pipeline uses it in three places:

1. `generate_{train,test}_segments`: segment lists determine the
   fetch jobs (`background-{start}-{duration}.hdf5` targets).
2. `compute_waveform_branches`: one testing-waveform job per
   (segment, shift multiple).
3. `compute_branch_map`: one inference job per
   (background file, shift multiple).

Each checkpoint writes a JSON branch map. After it completes,
Snakemake re-evaluates the DAG and the downstream `aggregate_*` rules
expand over the branch IDs. 


## Profiles

Profiles determine snakemake configuration:

- **`local`**: run everything as a subprocess on the local node
    for testing.
- **`condor`**: the LDG profile (name changed in PR #504). 
    Per-rule resources are set with `set-resources`. Two LDG
    specifics worth knowing:
  - SciTokens: every job receives `+OAuthServicesNeeded = scitokens`
    and accounting group attributes from `$ENV(LIGO_GROUP)` /
    `$ENV(LIGO_USERNAME)` via `classad_`-prefixed keys in
    `default-resources`.
  - Environment forwarding: environment variables are forwarded
    via `getenv`.

A Slurm profile is added in PR #504.

## Configuration

`pipeline/config/config.yaml` is the single source of pipeline
parameters. The intended workflow is to initialize a run directory 
with `aframe-init snakemake`, which creates a minimal config stub
and a copy of `train.yaml` for the run. Edit the stub to override any
parameters you want to change, and everything else will inherit from
`pipeline/config/config.yaml`. Individual values can also be overridden
at the command line with `--config key=value`.

## Data layout

```
{background_dir}/{train,test}/segments.txt
{background_dir}/{train,test}/background-{start}-{duration}.hdf5
{waveforms_dir}/train/{val_waveforms,training_waveforms}.hdf5
{waveforms_dir}/test/{waveforms,rejected_parameters}.hdf5
{run_dir}/{train,export,triton,infer,plots}/...
```

`background_dir` and `waveforms_dir` default to living under
`run_dir` but can be pointed at shared locations.

## To-dos

**Remote training.** The `remote_train: true` path submits a training
job to Nautilus. This hasn't been tested at all within Snakemake.

**HTCondor log organization.** HTCondor job logs land in
`.snakemake/htcondor/{clusterid}.{log,out,err}` with no rule
association.

**Hermes release and Volta deprecation.** The export and infer
packages pin hermes to `branch = "dev"` rather than a released
version. Cutting a hermes release will also drop support for Volta-era
GPUs.

**Hermes Triton image discovery.** `triton_image` in `config.yaml`
must currently be an explicit absolute path because the more modern
Triton containers have not been added to CVMFS. We could instead
have a shared cache directory to auto-pull from `ghcr.io/ml4gw/hermes`.

**Hyperparameter tuning currently has no Snakemake equivalent.** The law
pipeline had a `TuneTask` that stood up a Ray cluster on Kubernetes via
Helm and ran a distributed search. Nothing does that now; however,
`projects/train/configs/tune.yaml` and the `RayCluster` helm wrapper in
`projects/train/train/helm.py` are both preserved. The cluster sizing lived 
in the deleted `aframe/config.py` and was passed as Helm values by
`aframe/tasks/train/tune.py`. Recorded here so it is not lost:

```
head.cpu               32
head.memory            32G
worker.replicas        1
worker.gpu             2      (gpus_per_replica)
worker.cpu             12     per gpu, so 24 per replica
worker.memory          70G    per gpu, so 140G per replica
worker.min_gpu_memory  15000  MB
```
