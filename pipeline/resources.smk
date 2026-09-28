"""Resource helpers shared by all rules.

The slurm and htcondor executor plugins read different resource keys.
Slurm uses `mem_mb`, `runtime` and `gpu`, while htcondor ignores those
and reads `htcondor_request_mem_mb`, `allowed_execute_duration`, and
`request_gpus` plus its GPU matchmaking keys. Each helper returns both
slurm and htcondor sets.

Also resolves each project's container image.
"""

import os
import subprocess

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
