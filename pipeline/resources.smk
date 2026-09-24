"""Resource helpers shared by all rules.

The slurm and htcondor executor plugins read different resource keys.
Slurm uses `mem_mb`, `runtime` and `gpu`, while htcondor ignores those
and reads `htcondor_request_mem_mb`, `allowed_execute_duration`, and
`request_gpus` plus its GPU matchmaking keys. Each helper returns both
slurm and htcondor sets.
"""


def rule_resources(name):
    """Memory and walltime for rule `name`, from the config's `resources`.

    Anything a rule doesn't set there fall back to the profile's
    default-resources.
    """
    res = config.get("resources", {}).get(name, {})
    out = {}
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
