"""Snakemake rules for batch inference.

Rules:
  compute_branch_map  checkpoint: enumerate (file, shifts) inference branches
  start_triton        start the Triton server, record its IP
  infer_branch        run inference for one (file, shifts) branch
  aggregate_infer     merge per-branch outputs into final event sets
  stop_triton         signal the server to shut down


At very large branch counts, the DAG branch construction
can be limited:

    snakemake --batch aggregate_infer=1/10
"""

import json
import os
from pathlib import Path

import yaml

triton_dir = run_dir / "triton"
infer_dir = run_dir / "infer"
infer_log_dir = log_dir / "infer"
zero_lag = config.get("zero_lag", False)
return_timeseries = config.get("return_timeseries", False)

# How many concurrent inference sequences the Triton server can host.
# streams_per_gpu is the snapshotter's per-GPU instance count, set at export
num_gpus = len(str(config["gpus"]).split(","))
with open(config["export_config"]) as f:
    streams_per_gpu = yaml.safe_load(f).get("streams_per_gpu", 1)

# Each branch has two sequences, background and foreground, so the server can
# host streams_per_gpu * num_gpus / 2 branches at once. Register this as a
# global resource and have each infer_branch consume two streams.
workflow.global_resources["triton_streams"] = streams_per_gpu * num_gpus

# Per-branch request rate that holds aggregate load at rate_per_gpu * num_gpus.
# num_gpus cancels: more GPUs buy more concurrent branches, not faster ones.
rate_per_gpu = config.get("rate_per_gpu")
infer_rate = 2 * rate_per_gpu / streams_per_gpu if rate_per_gpu else "null"

INFER_CONTAINER = os.path.join(os.getenv("AFRAME_CONTAINER_ROOT", ""), "infer.sif")


localrules:
    compute_branch_map,
    start_triton,
    stop_triton,


wildcard_constraints:
    branch_id=r"\d+",


def _infer_branch_params(wildcards, input):
    with open(input.branch_map) as f:
        branch = json.load(f)[wildcards.branch_id]
    return {
        "fname": branch["fname"],
        "shifts": _fmt_list(branch["shifts"]),
    }


def get_infer_branch_files(wildcards):
    """Per-branch inference outputs."""
    bmap_file = checkpoints.compute_branch_map.get(**wildcards).output[0]
    with open(bmap_file) as f:
        branch_map = json.load(f)
    ids = list(branch_map.keys())
    files = {
        "background": expand(
            str(infer_dir / "tmp" / "{branch_id}" / "background.hdf5"), branch_id=ids
        ),
        "foreground": expand(
            str(infer_dir / "tmp" / "{branch_id}" / "foreground.hdf5"), branch_id=ids
        ),
        "metadata": expand(
            str(infer_dir / "tmp" / "{branch_id}" / "metadata.json"), branch_id=ids
        ),
    }
    if return_timeseries:
        files["timeseries"] = expand(
            str(infer_dir / "tmp" / "{branch_id}" / "timeseries.hdf5"), branch_id=ids
        )
    return files


checkpoint compute_branch_map:
    """Enumerate (background file, shifts) inference branches.

The number of shift multiples is the minimum needed to accumulate
Tb seconds of background livetime. Branches that are too short to
analyze after shifting and PSD burn-in are dropped. Optionally
includes zero-lag branches.
"""
    input:
        background=get_test_background_files,
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
        branch_map, i = {}, 0
        for fname, (start, stop) in zip(input.background, segments):
            if config.get("zero_lag", False):
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


rule infer_branch:
    """Run inference for one (background file, shifts) branch."""
    input:
        branch_map=str(infer_dir / "branch_map.json"),
        waveforms=str(test_waveforms / "waveforms.hdf5"),
        triton_started=str(triton_dir / "triton.started"),
    output:
        **(
            {"timeseries": str(infer_dir / "tmp" / "{branch_id}" / "timeseries.hdf5")}
            if return_timeseries
            else {}
        ),
        background=str(infer_dir / "tmp" / "{branch_id}" / "background.hdf5"),
        foreground=str(infer_dir / "tmp" / "{branch_id}" / "foreground.hdf5"),
        metadata=str(infer_dir / "tmp" / "{branch_id}" / "metadata.json"),
    log:
        str(infer_log_dir / "infer_branch-{branch_id}.log"),
    container:
        INFER_CONTAINER
    resources:
        triton_streams=2,
    params:
        branch=_infer_branch_params,
        ip_file=str(triton_dir / "triton.ip"),
        outdir=str(infer_dir / "tmp" / "{branch_id}"),
        ifos="[" + ",".join(config["ifos"]) + "]",
        model_name=config["model_name"],
        model_version=config["model_version"],
        inference_sampling_rate=config["inference_sampling_rate"],
        batch_size=config["inference_batch_size"],
        rate=infer_rate,
        return_timeseries=return_timeseries,
        psd_length=config["psd_length"],
        fduration=config["fduration"],
        integration_window_length=config["integration_window_length"],
        cluster_window_length=config["cluster_window_length"],
    shell:
        "infer"
        " --client.address $(cat {params.ip_file}):8001"
        " --client.model_name {params.model_name}"
        " --client.model_version {params.model_version}"
        " --data.background_fname {params.branch[fname]}"
        " --data.injection_set_fname {input.waveforms}"
        " '--data.ifos={params.ifos}'"
        " '--data.shifts={params.branch[shifts]}'"
        " --data.inference_sampling_rate {params.inference_sampling_rate}"
        " --data.batch_size {params.batch_size}"
        " --data.rate {params.rate}"
        " --postprocessor.psd_length {params.psd_length}"
        " --postprocessor.fduration {params.fduration}"
        " --postprocessor.integration_window_length"
        " {params.integration_window_length}"
        " --postprocessor.cluster_window_length"
        " {params.cluster_window_length}"
        " --return_timeseries {params.return_timeseries}"
        " --outdir {params.outdir}"
        " &> {log}"


rule aggregate_infer:
    """Merge per-branch outputs into the final background and foreground."""
    input:
        unpack(get_infer_branch_files),
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


rule stop_triton:
    """Shut down the Triton server by creating its stop sentinel."""
    input:
        background=str(infer_dir / "background.hdf5"),
    output:
        touch(str(triton_dir / "triton.stopped")),
    shell:
        "touch " + str(triton_dir / "triton.stop")
