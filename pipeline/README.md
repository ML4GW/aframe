# The Aframe Snakemake Pipeline

## Quick start

```bash
# One-time: create the orchestrator environment from its conda-lock file
conda-lock install --name aframe-smk pipeline/envs/snakemake.conda-lock.yml
conda activate aframe-smk

# One-time: set environment variables in your shell config
export AFRAME_CONTAINER_ROOT=~/aframe/images   # where the images are built
export LIGO_USERNAME=albert.einstein           # LDG accounting
export LIGO_GROUP=ligo.dev.o4.cbc.allsky.aframe
export WANDB_API_KEY=...                       # if the train config logs to W&B
# Optional: comma-separated extra directories to bind into containers on
# the submit node, beyond the run directory and those it reuses
export AFRAME_DATA_DIRS=...

# One-time: build the per-project containers
uv run build-containers # may need to build one at a time
# For the ldg-transfer profile, also publish the ones condor jobs run in,
# from a CIT AP. Repeat after changing an environment.
python -m scripts.publish_images

# One-time, for inference_mode: triton: pull the server image named by
# triton_image in config.yaml
apptainer pull $AFRAME_CONTAINER_ROOT/tritonserver_25.06.sif \
    docker://nvcr.io/nvidia/tritonserver:25.06-py3

# Initialize a run directory with a config.yaml, train.yaml, and run.sh
uv run aframe-init snakemake -d /path/to/my-run
# Edit /path/to/my-run/config.yaml to override parameters in pipeline/config/config.yaml
# Edit /path/to/my-run/train.yaml to adjust model/training hyperparameters.

# Arguments to run.sh go to snakemake, e.g. -n (dry-run) or --until <rule>.
cd /path/to/my-run
./run.sh
```

## Layout

```
Snakefile                     entry point: config + includes + rule all
pipeline/
  config/config.yaml          all pipeline parameters (defaults for every run)
  config/small.yaml           small end-to-end check preset
  config/review.yaml          short end-to-end validation preset
  profiles/ldg/               LDG HTCondor execution
  profiles/ldg-transfer/      LDG HTCondor execution, nothing shared with execute points
  profiles/delta/             Delta Slurm execution
  condor/job_wrapper.sh       executable of every ldg-transfer condor job
  resources.smk               containers, binds, per-rule resources
  envs/snakemake.yaml         orchestrator env
  envs/dev.yaml               local dev env
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

On the LDG, background fetching, waveform generation, `inprocess` inference
and sensitive volume run via condor. Training, export, the Triton server and
its clients, and the aggregation scripts run on the node where Snakemake is
launched (`gpu_rules_local` and `aggregate_rules_local`), and so this node
must have GPUs. The checkpoints, segment queries, PSDs and the `stop_triton`
rule also run on this node. On Delta, the `delta` profile sets both flags
false, so those rules are Slurm jobs too. For an all-local run, add
`--executor local --cores N --jobs N` to `run.sh`.

## Checkpoints

Most of the DAG cannot be determined up front: the number of
strain-fetching jobs depends on what segments DQSegDB returns, and the
number of waveform/inference branches depends on which segments are
long enough to analyze. Snakemake's equivalent is
the [checkpoint](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#data-dependent-conditional-execution)
mechanism, and the pipeline uses it in two places:

1. `generate_segments` (per split): segment lists determine the
   fetch jobs (`background-{start}-{duration}.hdf5` targets) and the
   testing-waveform branches.
2. `compute_branch_map`: one inference branch per
   (background file, timeslide), grouped into jobs of at most
   `branches_per_job`.

After each checkpoint completes, Snakemake re-evaluates the DAG and the
downstream rules expand over its output.

## Profiles

Profiles determine snakemake configuration:

- **`ldg`**: the LDG profile, for execute points that mount `/home`.
    Per-rule memory and walltime come from the run config's
    `resources` block (see `pipeline/resources.smk`). Two LDG
    specifics worth knowing:
  - SciTokens: every job receives `+OAuthServicesNeeded = scitokens`
    and accounting group attributes from `$ENV(LIGO_GROUP)` /
    `$ENV(LIGO_USERNAME)` via `classad_`-prefixed keys in
    `default-resources`.
  - Environment forwarding: environment variables are forwarded
    via `getenv`.

  At CIT, where most execute points no longer mount `/home`, set
  `epnfs: true` to match only those that do.
- **`ldg-transfer`**: the LDG profile for execute points that share
    nothing with the submit node (see below). Tokens and accounting are
    as for `ldg`, and environment variables are forwarded explicitly via
    `envvars`.
- **`delta`**: Slurm on Delta.

### Condor jobs without a shared filesystem

The `ldg-transfer` profile shares nothing with execute points
(`shared-fs-usage: none`). Each job runs snakemake for its rule inside the
project's published image, which condor fetches from OSDF. Condor sends the
job `code/` and the rule's declared inputs at their paths relative to the run
directory, and returns its declared outputs and `logs/`.
`pipeline/condor/job_wrapper.sh` puts `code/` first on `PYTHONPATH`, points
`HOME` at the job's scratch directory, and makes a failed job end with its
error rather than going on hold.

A rule that can run as a condor job must therefore:

- use `shell:`, not `script:`;
- name every file it reads or writes in `input`, `output` or `log`, never
  in a param, a glob or a directory convention;
- declare any checkpoint output that its own input functions read;
- get its software from its image.

Snakemake in the images is checked against the submit node's at startup,
since each job runs the image's snakemake with a command written by the
submit node's.

## Configuration

`pipeline/config/config.yaml` is the single source of pipeline
parameters. The intended workflow is to initialize a run directory 
with `aframe-init snakemake`, which creates a minimal config stub
and a copy of `train.yaml` for the run. Edit the stub to override any
parameters you want to change, and everything else will inherit from
`pipeline/config/config.yaml`. Individual values can also be overridden
at the command line with `--config key=value`. `--preset small` or
`--preset review` starts the stub from a preset, and `--profile` picks
the profile (default `pipeline/profiles/ldg`).

`run.sh` passes the default and run configs with `--configfiles`, in that
order.

## Data layout

All paths are relative to the run directory:

```
data/{train,test}/segments.txt
data/{train,test}/background-{start}-{duration}.hdf5
waveforms/train/{val_waveforms,training_waveforms}.hdf5
waveforms/test/{waveforms,rejected_parameters}.hdf5
{train,export,triton,infer,plots}/...
logs/
code/
```

`background_dir` and `waveforms_dir` name another run's `data/` or
`waveforms/` to use instead of generating new data. Their files are linked
into this run, and new files stay in this run's directory.

Each `./run.sh` copies the files git knows about into a read-only `code/`
and runs that copy, so edits to the local repo take effect when the run is
next started, and untracked files are left out (with a warning). To restart
a run, run `./run.sh` again. If snakemake was killed, remove its stale locks
first with `rm .snakemake/locks/*`.

## Containers

Each project is built into its own apptainer image by
`uv run build-containers`, which fills the templates in
`container_templates/` from the project's `pyproject.toml` and the
root `uv.lock`. Code always comes from the run's `code/` copy, which the
profiles bind over the image's copy at `/opt/aframe` (or, under
`ldg-transfer`, put first on `PYTHONPATH`), so an image only
needs rebuilding when its environment changes. Each image records a
hash of what shapes its environment (`scripts/env_hash.py`): its
project's and libraries' `pyproject.toml`s, its `apptainer.*` hooks, the
build template, the root `[tool.uv]` settings, and only the packages it
installs from `uv.lock` (as `uv export` lists them), so a lock change for
one project doesn't touch the others. Each image also
records the commit it was built from in `/opt/build_commit`, marked
`-dirty` if its environment files had uncommitted changes. The pipeline
warns at startup if a local image's hash doesn't match the local repo's.

**Published images.** The `data`, `infer` and `plots` images, the ones
condor jobs can run in, are published to OSDF staging as
`aframe-<project>-<hash>.sif`. Everyone publishes their own: build the
images, then on a CIT AP run `python -m scripts.publish_images`, which
copies them to `/igwn/cit/staging/<USER>`. Only images built from
committed environment files are published. Under the `ldg-transfer`
profile, condor jobs use the images published for the local repo's
environment. With `container_source: osdf`, the rules on the submit node
use them too, read through a CIT AP's `/osdf` mount. Either way, the
pipeline fails at startup if one hasn't been published. So publish again
after changing an environment, not after changing code.
`osdf_staging_dir` reads someone else's instead, since anyone's can be
read.

Jobs read images through OSDF caches, which keep serving the copy they
already hold even if the file at CIT is replaced, so a name must never be
reused for different contents. Naming images by hash guarantees that. Staging
has no documented quota or retention, so clean up your own as you go:
`rm /osdf/igwn/cit/staging/<USER>/aframe-*-<hash>.sif`, once no running
workflow uses that image.

Two more notes:

**`vizapp` is not containerized.** `plots.sif` covers only the
`sensitive-volume` rule, so it carries no torch at all, which
keeps the container small. The vizapp dependencies are instead 
in an extra, and the Bokeh app can be run locally with 
`uv sync --package plots --extra vizapp`, or `--extra vizapp-cuda` 
to run the model on a GPU, matching `device` in the vizapp config.

**Torch is put behind an extra in `data` and `plots`.** Both install it
from the PyTorch CPU index, which keeps CUDA dependencies out of the image. 
uv resolves each package once, so the CPU and CUDA builds can only 
coexist in the root `uv.lock` because each project declares its two 
variants as conflicting extras (`cpu`/`cuda`,`vizapp`/`vizapp-cuda`), 
which uv then resolves separately. As a result, one extra must always
be installed for these projects. For `data`, this should be the `cpu`
extra, as the `cuda` extra doesn't do anything.

## To-dos

**Remote training.** The `remote_train: true` path submits a training
job to Nautilus. This hasn't been tested at all within Snakemake.

**HTCondor log organization.** HTCondor job logs land in
`.snakemake/htcondor/{clusterid}.{log,out,err}` with no rule
association.

**Dependencies tracked from branches.** `export` and `infer` pin
hermes to `ML4GW/hermes` branch `dev`, and `online` pins amplfi to
`ML4GW/amplfi` branch `main`, rather than released versions. Both
should move to releases once possible, though note that a hermes
release will also drop support for Volta-era GPUs. 

**Blocked on moving to CUDA 13.** Two conditions have to
be met before we can move to CUDA 13:

1. **Dropping Volta (V100) support.** CUDA 13 drops Volta (`sm_70`)
   and Pascal (`sm_60`), so these are held at CUDA 12 to keep V100
   nodes on the LDG usable.
2. **Driver >= 580 everywhere we run.** CUDA 13.x requires NVIDIA
   driver >= 580. Delta is on 570, so it cannot run CUDA 13 binaries,
   regardless of GPU architecture.

If these conditions are met, then we can implement various upgrades:

- `constraint-dependencies = ["torchaudio==2.10.0"]` in the root
  `pyproject.toml`. torchaudio 2.11 is a CUDA 13 build.
- `torch==2.10.0` in `train`, `export`, `infer` and `online`.
- `cuda-minimal-build-12-8` and NVIDIA's `debian12` apt repo in
  `projects/infer/apptainer.post`.
- The uv base image in `container_templates/uv.def`, pinned to
  `0.9.30-python3.12-bookworm-slim`.
- Moving hermes from the `dev` branch to a release, per the note above.
- `nvidia-cudnn-cu12` and `tensorrt-cu12==10.11.0.33` in
  `projects/export`, plus the Triton image in `config.yaml`.

**Hermes Triton image discovery.** `triton_image` in `config.yaml`
must currently name a local image (a bare filename resolved against
`$AFRAME_CONTAINER_ROOT`, or an absolute path) because the more modern
Triton containers have not been added to CVMFS. We could instead
have a shared cache directory to auto-pull from
`nvcr.io/nvidia/tritonserver:<release>-py3`. (hermes' images on
`ghcr.io/ml4gw/hermes` add only a label and stop at 24.12.)

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
