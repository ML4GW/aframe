"""Resource helpers shared by all rules.

The slurm and htcondor executor plugins read different resource keys.
Slurm uses `mem_mb`, `runtime` and `gpu`, while htcondor ignores those
and reads `htcondor_request_mem_mb`, `allowed_execute_duration`, and
`request_gpus` plus its GPU matchmaking keys. Each helper returns both
slurm and htcondor sets.

Also resolves each project's container image.
"""

import os
import shutil
import subprocess

from snakemake.logging import logger

from scripts.env_hash import env_hash


def container(project):
    """The image to run `project`'s rules in:
    `$AFRAME_CONTAINER_ROOT/<project>.sif`.
    """
    return os.path.join(os.getenv("AFRAME_CONTAINER_ROOT", ""), f"{project}.sif")


def check_images(projects=("data", "train", "export", "infer", "plots")):
    """Warn about images built for a different environment than the local
    repo's.

    Images record their environment hash (scripts/env_hash.py) at build time.
    """
    exe = shutil.which("apptainer") or shutil.which("singularity")
    if exe is None:
        return
    for project in projects:
        image = container(project)
        if not os.path.exists(image):
            continue
        result = subprocess.run(
            [exe, "exec", image, "cat", "/opt/env_hash"],
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
    res = config.get("resources", {}).get(name, {})
    out = {}
    if config.get("epnfs"):
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
        "slurm_partition": config.get("inference_partition", "gpuA40x4"),
        "gpu": 1,
        "request_gpus": 1,
    }
    if config.get("inference_backend", "export") == "aoti":
        res["require_gpus"] = f"Capability == {config['aoti_gpu_capability']}"
    else:
        res["gpus_minimum_capability"] = config["gpu_min_capability"]
        if config.get("gpu_min_memory_mb"):
            res["gpus_minimum_memory"] = config["gpu_min_memory_mb"]
    return res
