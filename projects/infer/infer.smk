"""Snakemake rules for batch inference.

Two inference modes, selected by `inference_mode` in the run config.
Both run branches in groups of size `branches_per_job` via
`infer-triton` / `infer-local`, and both write the same outputs.
The `compute_branch_map` and `aggregate_infer` rules are shared.

    triton: A Triton server hosts the model, and each group job is a CPU client
            that streams its branches to it. Best when GPUs are scarce but
            multiple exist on a node. Realistically, only used on LDG.

    inprocess: Each group job is one GPU job that loads the model locally.
               Best when there are many GPUs available and jobs can be
               scheduled on them; e.g., Delta or OSG.

At very large branch counts the DAG construction can be batched:
    snakemake --batch aggregate_infer=1/10
"""

import json
import math
import os
from pathlib import Path

triton_dir = run_dir / "triton"
infer_dir = run_dir / "infer"
infer_log_dir = log_dir / "infer"
zero_lag = config.get("zero_lag", False)
return_timeseries = config.get("return_timeseries", False)

INFERENCE_MODE = config.get("inference_mode", "triton")
INFERENCE_BACKEND = config.get("inference_backend", "export")
if INFERENCE_MODE not in ("triton", "inprocess"):
    raise WorkflowError(
        f"inference_mode must be 'triton' or 'inprocess', got {INFERENCE_MODE}"
    )
if INFERENCE_BACKEND not in ("export", "compile", "aoti"):
    raise WorkflowError(
        f"inference_backend must be 'export', 'compile', or 'aoti', got {INFERENCE_BACKEND}"
    )

INFER_CONTAINER = os.path.join(os.getenv("AFRAME_CONTAINER_ROOT", ""), "infer.sif")

AOTI_PKG = str(export_out / "model_aoti.pt2")

BRANCHES_PER_JOB = config.get("branches_per_job", 1)


wildcard_constraints:
    branch_id=r"\d+",
    group_id=r"\d+",


def _num_groups(branch_map_file):
    with open(branch_map_file) as f:
        n = len(json.load(f))
    return math.ceil(n / BRANCHES_PER_JOB)


def get_infer_group_sentinels(wildcards):
    """One sentinel per group of branches_per_job branches"""
    bmap_file = checkpoints.compute_branch_map.get(**wildcards).output[0]
    return {
        "sentinels": expand(
            str(infer_dir / "tmp" / "groups" / "{group_id}.done"),
            group_id=list(range(_num_groups(bmap_file))),
        )
    }


checkpoint compute_branch_map:
    """Enumerate (background file, shifts) inference branches.

The number of shift multiples is the minimum needed to accumulate
Tb seconds of background livetime. Branches that are too short to
analyze after shifting and PSD burn-in are dropped. Optionally
includes zero-lag branches.
"""
    input:
        background=get_test_background_files,
        waveform_branch_map=str(test_waveforms / "waveform_branch_map.json"),
    output:
        str(infer_dir / "branch_map.json"),
    run:
        def _get_num_shifts(segments, Tb, shift, psd_length):
            if Tb == 0:
                return 0
            livetime, num_shifts = 0, 0
            durations = [stop - start - psd_length for start, stop in segments]
            while livetime < Tb:
                num_shifts += 1
                for dur in durations:
                    dur -= shift * num_shifts
                    if dur > 0:
                        livetime += dur
            return num_shifts

        shifts = config["shifts"]
        psd_length = config["psd_length"]
        segments = []
        for fname in input.background:
            start, duration = map(float, Path(fname).stem.split("-")[-2:])
            segments.append((start, start + duration))
        num_shifts = _get_num_shifts(
            segments, config["Tb"], max(shifts), psd_length
        )
        with open(input.waveform_branch_map) as f:
            wbmap = json.load(f)
        max_waveform_shift = max(max(b["shifts"]) for b in wbmap.values())
        num_waveform_shifts = max_waveform_shift / max(shifts)
        if num_waveform_shifts > num_shifts:
            raise WorkflowError(
                f"num_testing_signals requires {num_waveform_shifts} shift "
                f"multiples but Tb={config['Tb']} only covers {num_shifts}. "
                f"Reduce num_testing_signals or increase Tb."
            )
        branch_map, i = {}, 0
        for fname, (start, stop) in zip(input.background, segments):
            if zero_lag:
                zero_shifts = [0] * len(shifts)
                if _is_analyzeable_segment(start, stop, zero_shifts, psd_length):
                    branch_map[str(i)] = {
                        "fname": str(fname),
                        "shifts": zero_shifts,
                    }
                    i += 1
            for j in range(num_shifts):
                shift = [(j + 1) * s for s in shifts]
                if _is_analyzeable_segment(start, stop, shift, psd_length):
                    branch_map[str(i)] = {
                        "fname": str(fname),
                        "shifts": shift,
                    }
                    i += 1
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        with open(output[0], "w") as f:
            json.dump(branch_map, f, indent=2)


_group_common_params = dict(
    branches_per_job=BRANCHES_PER_JOB,
    outdir_root=str(infer_dir / "tmp"),
    ifos="[" + ",".join(config["ifos"]) + "]",
    inference_sampling_rate=config["inference_sampling_rate"],
    batch_size=config["inference_batch_size"],
    psd_length=config["psd_length"],
    fduration=config["fduration"],
    integration_window_length=config["integration_window_length"],
    cluster_window_length=config["cluster_window_length"],
    return_timeseries=return_timeseries,
)

_GROUP_SHELL_SUFFIX = (
    " --branch_map {input.branch_map}"
    " --group_id {wildcards.group_id}"
    " --branches_per_job {params.branches_per_job}"
    " --waveforms {input.waveforms}"
    " --outdir_root {params.outdir_root}"
    " '--ifos={params.ifos}'"
    " --inference_sampling_rate {params.inference_sampling_rate}"
    " --batch_size {params.batch_size}"
    " --psd_length {params.psd_length}"
    " --fduration {params.fduration}"
    " --integration_window_length {params.integration_window_length}"
    " --cluster_window_length {params.cluster_window_length}"
    " --return_timeseries {params.return_timeseries}"
    " &> {log}"
)


if INFERENCE_MODE == "triton":

    num_gpus = len(str(config["gpus"]).split(","))
    streams_per_gpu = config["streams_per_gpu"]

    workflow.global_resources["triton_streams"] = streams_per_gpu * num_gpus
    rate_per_gpu = config.get("rate_per_gpu")
    infer_rate = 2 * rate_per_gpu / streams_per_gpu if rate_per_gpu else "null"

    localrules:
        compute_branch_map,
        start_triton,
        stop_triton,

    rule start_triton:
        """Start the Triton inference server on the submit node."""
        input:
            model_repo=str(export_out / "model_repo"),
        output:
            str(triton_dir / "triton.started"),
        log:
            str(infer_log_dir / "start_triton.log"),
        params:
            output_dir=str(triton_dir),
            ip_file=str(triton_dir / "triton.ip"),
            stop_sentinel=str(triton_dir / "triton.stop"),
            logfile=str(triton_dir / "server.log"),
            model_name=config["model_name"],
            model_version=config["model_version"],
            gpus=config["gpus"],
            batch_size=config["inference_batch_size"],
            triton_image=config["triton_image"],
            idle_timeout=config.get("triton_idle_timeout", 3600),
        script:
            "scripts/start_triton.py"

    rule infer_group:
        """Stream a group of branches to the Triton server from one CPU client."""
        input:
            branch_map=str(infer_dir / "branch_map.json"),
            waveforms=str(test_waveforms / "waveforms.hdf5"),
            triton_started=str(triton_dir / "triton.started"),
        output:
            touch(str(infer_dir / "tmp" / "groups" / "{group_id}.done")),
        log:
            str(infer_log_dir / "infer_group-{group_id}.log"),
        container:
            INFER_CONTAINER
        resources:
            triton_streams=2,
        params:
            **_group_common_params,
            ip_file=str(triton_dir / "triton.ip"),
            model_name=config["model_name"],
            model_version=config["model_version"],
            rate=infer_rate,
        shell:
            "infer-triton"
            " --address $(cat {params.ip_file}):8001"
            " --model_name {params.model_name}"
            " --model_version {params.model_version}"
            " --rate {params.rate}" + _GROUP_SHELL_SUFFIX

    rule stop_triton:
        """Shut down the Triton server by creating its stop sentinel."""
        input:
            background=str(infer_dir / "background.hdf5"),
        output:
            touch(str(triton_dir / "triton.stopped")),
        shell:
            "touch " + str(triton_dir / "triton.stop")

else:

    localrules:
        compute_branch_map,

    if INFERENCE_BACKEND == "aoti":
        _artifact = AOTI_PKG

        rule compile_model:
            """Compile model_exported.pt2 to an AOTInductor package."""
            input:
                exported=str(train_out / "model_exported.pt2"),
            output:
                AOTI_PKG,
            log:
                str(infer_log_dir / "compile_model.log"),
            container:
                INFER_CONTAINER
            resources:
                slurm_partition=config.get("inference_partition", "gpuA40x4"),
                gpu=1,
                mem_mb=config.get("compile_mem_mb", 32000),
                runtime=10,
            params:
                num_ifos=len(config["ifos"]),
                sample_rate=config["sample_rate"],
                kernel_length=config["kernel_length"],
                batch_size=config["inference_batch_size"],
            script:
                "../export/scripts/compile_aoti.py"

    else:
        _artifact = str(train_out / "model_exported.pt2")

    rule infer_group:
        """Run a group of branches in-process on one GPU."""
        input:
            branch_map=str(infer_dir / "branch_map.json"),
            waveforms=str(test_waveforms / "waveforms.hdf5"),
            artifact=_artifact,
        output:
            touch(str(infer_dir / "tmp" / "groups" / "{group_id}.done")),
        log:
            str(infer_log_dir / "infer_group-{group_id}.log"),
        container:
            INFER_CONTAINER
        resources:
            slurm_partition=config.get("inference_partition", "gpuA40x4"),
            gpu=1,  # slurm
            request_gpus=1,  # condor
            mem_mb=config.get("infer_mem_mb", 32000),
            runtime=config.get("infer_runtime", 60),
        params:
            **_group_common_params,
            weights=_artifact,
            backend=INFERENCE_BACKEND,
            aoti_arg=(f" --aoti_path {AOTI_PKG}" if INFERENCE_BACKEND == "aoti" else ""),
            sample_rate=config["sample_rate"],
            kernel_length=config["kernel_length"],
            highpass=config["highpass"],
            fftlength=config.get("fftlength") or "null",
        shell:
            "infer-local"
            " --weights {params.weights}"
            " --backend {params.backend}{params.aoti_arg}"
            " --sample_rate {params.sample_rate}"
            " --kernel_length {params.kernel_length}"
            " --highpass {params.highpass}"
            " --fftlength {params.fftlength}" + _GROUP_SHELL_SUFFIX


rule aggregate_infer:
    """Merge per-branch outputs into the final background and foreground."""
    input:
        unpack(get_infer_group_sentinels),
        branch_map=str(infer_dir / "branch_map.json"),
    output:
        **({"zero_lag": str(infer_dir / "0lag.hdf5")} if zero_lag else {}),
        **(
            {"timeseries": str(infer_dir / "timeseries.hdf5")}
            if return_timeseries
            else {}
        ),
        background=str(infer_dir / "background.hdf5"),
        foreground=str(infer_dir / "foreground.hdf5"),
    log:
        str(infer_log_dir / "aggregate_infer.log"),
    container:
        INFER_CONTAINER
    params:
        tmp_dir=str(infer_dir / "tmp"),
    script:
        "scripts/aggregate_infer.py"
