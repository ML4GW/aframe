"""Resource helpers shared by all rules.

The slurm and htcondor executor plugins read different resource keys.
Slurm uses `mem_mb`, `runtime` and `gpu`, while htcondor ignores those
and reads `htcondor_request_mem_mb`, `allowed_execute_duration`, and
`request_gpus` plus its GPU matchmaking keys. Each helper returns both
slurm and htcondor sets.

Also resolves each project's container image, checks the image hash
against the source code's, and sets up run's directories to be
bound into the image.
"""

import os
import subprocess
from pathlib import Path

from snakemake.exceptions import WorkflowError
from snakemake.logging import logger

from scripts.env_hash import env_hash
from scripts.publish_images import OSDF_PROJECTS, image_name, staging_dir


def container(project):
    """The image that `project`'s rules run in.

    By default, the locally built `$AFRAME_CONTAINER_ROOT/<project>.sif`.
    With `container_source: osdf`, the data, infer and plots rules instead
    use the image published for the local repo's environment, read from
    `osdf_staging_dir` (your own staging directory by default) through the
    AP's `/osdf` mount.
    """
    if config["container_source"] == "osdf" and project in OSDF_PROJECTS:
        name = image_name(project, env_hash(project))
        source = config["osdf_staging_dir"] or staging_dir()
        return f"/osdf{source}/{name}"
    return os.path.join(os.getenv("AFRAME_CONTAINER_ROOT", ""), f"{project}.sif")


def set_container_binds():
    """Create the run's directories and set what the profiles'
    `apptainer-args` bind into each container.

    `AFRAME_REPO` is the local repo, which is bound over the code in the image.
    `AFRAME_DATA_DIRS` is the run directory, the directories whose files it
    reuses (where its links point) and any `rnp_frame_dir`, plus whatever the
    variable already held.
    """
    # REPO is defined in Snakefile
    os.environ["AFRAME_REPO"] = str(REPO)
    for path in (run_dir / "data", run_dir / "waveforms", log_dir):
        os.makedirs(path, exist_ok=True)
    dirs = os.getenv("AFRAME_DATA_DIRS", "").split(",") + [str(run_dir)]
    for key in ("background_dir", "waveforms_dir", "rnp_frame_dir"):
        if config.get(key):
            dirs.append(config[key])

    # Absolute paths, sorted by their number of parts
    paths = sorted(
        {Path(os.path.abspath(d)) for d in dirs if d}, key=lambda p: len(p.parts)
    )
    binds = []
    for path in paths:
        # Skip if the parent will be bound
        if not any(path.is_relative_to(parent) for parent in binds):
            binds.append(path)
    os.environ["AFRAME_DATA_DIRS"] = ",".join(str(path) for path in binds)
    logger.info(f"Binding {os.environ['AFRAME_DATA_DIRS']} into containers")


def link_files(source, dest):
    """Link each file under `source` to the same relative path under `dest`,
    unless something is already there.

    Condor can't send a job a file through a linked directory, so reused
    directories are linked file by file. Snakemake compares the links' own
    times, not their files', so each link gets its file's modification
    time, or a file linked after its inputs would look newer and be redone.
    """
    source = Path(source)
    for path in source.rglob("*"):
        if path.is_file():
            link = Path(dest) / path.relative_to(source)
            if not link.exists():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(path.resolve())
                stat = path.stat()
                os.utime(
                    link,
                    ns=(stat.st_atime_ns, stat.st_mtime_ns),
                    follow_symlinks=False,
                )


def check_images(projects=("data", "train", "export", "infer", "plots")):
    """Fail on missing published images, and warn about local images built
    for a different environment than the local repo's.

    Images record their environment hash (scripts/env_hash.py) at build time.
    """
    for project in projects:
        image = container(project)
        if image.startswith("/osdf/"):
            # published images are named by their environment hash
            if not os.path.exists(image):
                raise WorkflowError(
                    f"{image} doesn't exist. Either the local repo's "
                    f"{project} environment hasn't been published (build the "
                    "image, then run `python -m scripts.publish_images "
                    f"{project}` on a CIT AP), or this isn't a CIT AP. "
                    "Otherwise, set `container_source: local`."
                )
            continue
        if not os.path.exists(image):
            continue
        result = subprocess.run(
            ["apptainer", "exec", image, "cat", "/opt/env_hash"],
            capture_output=True,
            text=True,
        )
        built = result.stdout.strip() if result.returncode == 0 else "unknown"
        expected = env_hash(project)
        if built != expected:
            logger.warning(
                f"{image} was built for environment {built}, but the "
                f"local repo's is {expected}. Rebuild it with "
                f"`uv run build-containers {project}`."
            )


def gpu_env(gpus, num_gpus):
    """Shell command that sets CUDA_VISIBLE_DEVICES for a rule. Either uses
    `gpus` as given, or picks the `num_gpus` least-used GPUs with `auto` via
    scripts/free_gpus.py, Empty for null, which uses all visible GPUs.
    """
    if gpus is None:
        return ""
    if gpus == "auto":
        gpus = f"$(python /opt/aframe/scripts/free_gpus.py {num_gpus})"
    return f"CUDA_VISIBLE_DEVICES={gpus}; export CUDA_VISIBLE_DEVICES; "


def check_gpus():
    """Check that pinned GPU lists name as many GPUs as their counts."""
    for prefix in ("train", "inference"):
        gpus = config[f"{prefix}_gpus"]
        num_gpus = config[f"{prefix}_num_gpus"]
        if gpus is None or gpus == "auto":
            continue
        if len(str(gpus).split(",")) != num_gpus:
            raise WorkflowError(
                f"{prefix}_gpus ({gpus}) should list {prefix}_num_gpus "
                f"({num_gpus}) GPUs"
            )


def rule_resources(name):
    """Memory and walltime for rule `name`, from the config's `resources`.

    Anything a rule doesn't set there fall back to the profile's
    default-resources. With `epnfs`, condor jobs only match execute points
    that mount the AP's /home.
    """
    res = config["resources"].get(name, {})
    out = {}
    if config["epnfs"]:
        out["requirements"] = "TARGET.EPNFS =?= True"
    if "mem_mb" in res:
        out["mem_mb"] = out["htcondor_request_mem_mb"] = res["mem_mb"]
    if "runtime" in res:
        out["runtime"] = res["runtime"]  # minutes
        out["allowed_execute_duration"] = int(res["runtime"] * 60)  # seconds
    return out


def gpu_resources():
    """One GPU for an in-process inference rule, under slurm or condor.

    On condor, match any GPU above the configured floors. An AOTI package
    only runs on the architecture it was compiled for, so with the aoti
    backend pin compile_model and infer_group to one exact capability
    instead.
    """
    res = {
        "slurm_partition": config["inference_partition"],
        "gpu": 1,
        "request_gpus": 1,
    }
    if config["inference_backend"] == "aoti":
        res["require_gpus"] = f"Capability == {config['aoti_gpu_capability']}"
    else:
        res["gpus_minimum_capability"] = config["gpu_min_capability"]
        if config["gpu_min_memory_mb"]:
            res["gpus_minimum_memory"] = f"{config['gpu_min_memory_mb']}M"
    return res
