"""Snakemake rules for batch inference.

Two inference modes, selected by `inference_mode` in the run config.
Both run branches in groups of at most `branches_per_job` via
`infer-triton` / `infer-local`, and both write the same outputs.
The `compute_branch_map` and `aggregate_infer` rules are shared.

    triton: A Triton server on the submit node hosts the model, and each group
            job is a CPU client, also on the submit node, that streams its
            branches to it. Best when GPUs are scarce but multiple exist on
            the submit node. Realistically, only used on LDG.

    inprocess: Each group job is one GPU job that loads the model locally.
               Best when there are many GPUs available and jobs can be
               scheduled on them; e.g., Delta or OSG.

At very large branch counts the DAG construction can be batched:
    snakemake --batch aggregate_infer=1/10
"""

import json
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

ANALYSIS_TYPE = config.get("analysis_type", "hdf5")
if ANALYSIS_TYPE not in ("hdf5", "rnp"):
    raise WorkflowError(f"analysis_type must be 'hdf5' or 'rnp', got {ANALYSIS_TYPE}")
if ANALYSIS_TYPE == "rnp" and INFERENCE_MODE == "triton":
    raise WorkflowError("analysis_type 'rnp' requires inference_mode 'inprocess'")

INFER_CONTAINER = os.path.join(os.getenv("AFRAME_CONTAINER_ROOT", ""), "infer.sif")

TRITON_IMAGE = os.path.join(
    os.getenv("AFRAME_CONTAINER_ROOT", ""), config["triton_image"]
)


def check_triton_image():
    if INFERENCE_MODE == "triton" and not os.path.exists(TRITON_IMAGE):
        raise WorkflowError(
            f"Triton image {TRITON_IMAGE} doesn't exist. Pull the release "
            f"in its name, e.g. `apptainer pull {TRITON_IMAGE} "
            "docker://nvcr.io/nvidia/tritonserver:25.06-py3`."
        )


AOTI_PKG = str(export_out / "model_aoti.pt2")

BRANCHES_PER_JOB = config.get("branches_per_job", 1)


if ANALYSIS_TYPE == "rnp":
    FRAME_DIR = config["rnp_frame_dir"]
    CHANNEL = config["rnp_channel"]


wildcard_constraints:
    branch_id=r"\d+",
    group_id=r"\d+",


# Outputs of each group job, merged across groups by aggregate_infer.
# metadata.json holds the row count of each event set.
group_dir = infer_dir / "groups" / "{group_id}"
_group_outputs = {
    name: str(group_dir / f"{name}.hdf5")
    for name in ["background", "foreground"]
    + (["zero_lag"] if zero_lag else [])
    + (["timeseries"] if return_timeseries else [])
}
_group_outputs["metadata"] = str(group_dir / "metadata.json")


def _load_branch_map():
    bmap_file = checkpoints.compute_branch_map.get().output[0]
    with open(bmap_file) as f:
        return json.load(f)


def _assign_groups(branch_map, split_at_file_changes):
    """Number the branches, in order, into groups of at most branches_per_job.

    With `split_at_file_changes`, a group never spans two strain files, so
    a file is transferred to a job once for all of its shifts in that group.
    """
    group, size, last = -1, BRANCHES_PER_JOB, None
    for branch in branch_map.values():
        if size == BRANCHES_PER_JOB or (
            split_at_file_changes and branch["fname"] != last
        ):
            group, size = group + 1, 0
        branch["group"] = group
        size += 1
        last = branch["fname"]


def _group_branches(branch_map, group_id):
    return [b for b in branch_map.values() if b["group"] == group_id]


def get_infer_group_inputs(wildcards):
    """Strain files, plus their testing waveforms, for one group."""
    branches = _group_branches(_load_branch_map(), int(wildcards.group_id))
    inputs = {"background": sorted({b["fname"] for b in branches})}
    if ANALYSIS_TYPE == "hdf5":
        inputs["waveforms"] = [b["waveforms"] for b in branches if b["waveforms"]]
    return inputs


def get_infer_group_outputs(wildcards):
    """Every group's outputs, keyed by output name."""
    num_groups = 1 + max(b["group"] for b in _load_branch_map().values())
    return {
        name: expand(fname, group_id=range(num_groups))
        for name, fname in _group_outputs.items()
    }


if ANALYSIS_TYPE == "rnp":

    checkpoint compute_branch_map:
        """Enumerate R&P frame files as one branch each."""
        output:
            str(infer_dir / "branch_map.json"),
        run:
            files = sorted(str(p) for p in Path(FRAME_DIR).rglob("*.gwf"))
            if not files:
                raise WorkflowError(f"No .gwf frames found in {FRAME_DIR}")
            shifts = [0.0] * len(config["ifos"])
            branch_map = {
                str(i): {"fname": f, "shifts": shifts} for i, f in enumerate(files)
            }
            # one frame per branch, so there's nothing to share within a group
            _assign_groups(branch_map, split_at_file_changes=False)
            Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
            with open(output[0], "w") as f:
                json.dump(branch_map, f, indent=2)

else:

    checkpoint compute_branch_map:
        """Enumerate (background file, shifts) inference branches.

        The number of shift multiples is the minimum needed to accumulate
        Tb seconds of background livetime. Branches that are too short to
        analyze after shifting and PSD burn-in are dropped. Optionally
        includes zero-lag branches. Each branch records the waveform file of
        the testing waveform branch with the same file and shifts, which
        holds all of its injections, or null if there is none (e.g. zero-lag).
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
            wbranch_ids = {
                (w["background"], tuple(w["shifts"])): i for i, w in wbmap.items()
            }
            max_waveform_shift = max(max(b["shifts"]) for b in wbmap.values())
            num_waveform_shifts = max_waveform_shift / max(shifts)
            if num_waveform_shifts > num_shifts:
                raise WorkflowError(
                    f"num_testing_signals requires {num_waveform_shifts} shift "
                    f"multiples but Tb={config['Tb']} only covers {num_shifts}. "
                    f"Reduce num_testing_signals or increase Tb."
                )
            branch_shifts = [
                [(j + 1) * s for s in shifts] for j in range(num_shifts)
            ]
            if zero_lag:
                branch_shifts.insert(0, [0] * len(shifts))
            branch_map, i, matched = {}, 0, set()
            for fname, (start, stop) in zip(input.background, segments):
                for shift in branch_shifts:
                    if _is_analyzeable_segment(start, stop, shift, psd_length):
                        branch_map[str(i)] = {
                            "fname": str(fname),
                            "shifts": shift,
                            "waveforms": None,
                        }
                        wbranch_id = wbranch_ids.get((str(fname), tuple(shift)))
                        if wbranch_id is not None:
                            branch_map[str(i)]["waveforms"] = str(
                                test_waveforms
                                / "branches"
                                / wbranch_id
                                / "waveforms.hdf5"
                            )
                            matched.add(wbranch_id)
                        i += 1
            # every testing waveform must be analyzed by some branch
            unmatched = set(wbmap) - matched
            if unmatched:
                raise WorkflowError(
                    f"Testing waveform branches {sorted(unmatched, key=int)} "
                    "match no inference branch"
                )
            _assign_groups(branch_map, split_at_file_changes=True)
            Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
            with open(output[0], "w") as f:
                json.dump(branch_map, f, indent=2)


_group_common_params = dict(
    analysis_type=ANALYSIS_TYPE,
    ifos="[" + ",".join(config["ifos"]) + "]",
    inference_sampling_rate=config["inference_sampling_rate"],
    batch_size=config["inference_batch_size"],
    psd_length=config["psd_length"],
    fduration=config["fduration"],
    integration_window_length=config["integration_window_length"],
    cluster_window_length=config["cluster_window_length"],
)

_GROUP_SHELL_SUFFIX = (
    " --branch_map {input.branch_map}"
    " --group_id {wildcards.group_id}"
    " --analysis_type {params.analysis_type}"
    " --background_out {output.background}"
    " --foreground_out {output.foreground}"
    " --metadata_out {output.metadata}"
    + (" --zero_lag_out {output.zero_lag}" if zero_lag else "")
    + (" --timeseries_out {output.timeseries}" if return_timeseries else "")
    + " '--ifos={params.ifos}'"
    " --inference_sampling_rate {params.inference_sampling_rate}"
    " --batch_size {params.batch_size}"
    " --psd_length {params.psd_length}"
    " --fduration {params.fduration}"
    " --integration_window_length {params.integration_window_length}"
    " --cluster_window_length {params.cluster_window_length}"
    " &> {log}"
)


if INFERENCE_MODE == "triton":

    num_gpus = len(str(config["gpus"]).split(","))
    streams_per_gpu = config["streams_per_gpu"]

    workflow.global_resources["triton_streams"] = streams_per_gpu * num_gpus
    rate_per_gpu = config.get("rate_per_gpu")
    infer_rate = 2 * rate_per_gpu / streams_per_gpu if rate_per_gpu else "null"

    # The clients run where the server does because we can't
    # guarantee that the EP can reach the submit node.
    localrules:
        compute_branch_map,
        start_triton,
        infer_group,
        stop_triton,

    rule start_triton:
        """Start the Triton inference server on the submit node."""
        input:
            model_repo=str(export_out / "model_repo"),
        output:
            started=str(triton_dir / "triton.started"),
            ip_file=str(triton_dir / "triton.ip"),
        log:
            str(infer_log_dir / "start_triton.log"),
        params:
            output_dir=str(triton_dir),
            stop_sentinel=str(triton_dir / "triton.stop"),
            logfile=str(triton_dir / "server.log"),
            model_name=config["model_name"],
            model_version=config["model_version"],
            gpus=config["gpus"],
            batch_size=config["inference_batch_size"],
            triton_image=TRITON_IMAGE,
            idle_timeout=config.get("triton_idle_timeout", 3600),
        script:
            "scripts/start_triton.py"

    rule infer_group:
        """Stream a group of branches to the Triton server from a local client."""
        input:
            unpack(get_infer_group_inputs),
            branch_map=str(infer_dir / "branch_map.json"),
            triton_started=str(triton_dir / "triton.started"),
        output:
            **_group_outputs,
        log:
            str(infer_log_dir / "infer_group-{group_id}.log"),
        container:
            INFER_CONTAINER
        resources:
            triton_streams=2,
        params:
            **_group_common_params,
            model_name=config["model_name"],
            model_version=config["model_version"],
            rate=infer_rate,
        shell:
            "infer-triton"
            # a localrule, so the server is on this node
            " --address localhost:8001"
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
                **rule_resources("compile_model"),
                **gpu_resources(),
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
            unpack(get_infer_group_inputs),
            branch_map=str(infer_dir / "branch_map.json"),
            artifact=_artifact,
        output:
            **_group_outputs,
        log:
            str(infer_log_dir / "infer_group-{group_id}.log"),
        container:
            INFER_CONTAINER
        resources:
            **rule_resources("infer_group"),
            **gpu_resources(),
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
            " --fftlength {params.fftlength}"
            + (f" --channel {CHANNEL}" if ANALYSIS_TYPE == "rnp" else "")
            + _GROUP_SHELL_SUFFIX


rule aggregate_infer:
    """Merge per-group outputs into the final background and foreground."""
    input:
        unpack(get_infer_group_outputs),
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
    resources:
        **rule_resources("aggregate_infer"),
    params:
        analysis_type=ANALYSIS_TYPE,
    script:
        "scripts/aggregate_infer.py"
