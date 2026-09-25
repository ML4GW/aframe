"""Snakemake rules for data acquisition and waveform generation.

Rules:
  generate_train_segments       checkpoint: query DQSegDB for train segments
  generate_test_segments        checkpoint: query DQSegDB for test segments
  fetch_train_background        download one chunk of train strain data
  fetch_test_background         download one chunk of test strain data
  compute_waveform_branches     checkpoint: enumerate (segment, shifts) combos
  val_waveforms_branch          validation waveforms for one branch
  aggregate_val_waveforms       merge per-branch validation waveforms
  testing_waveforms_branch      testing waveforms for one branch
  aggregate_testing_waveforms   merge per-branch testing waveforms
  training_waveforms_branch     one branch of training waveform polarizations (optional)
  aggregate_training_waveforms  merge per-branch training waveforms (optional)

Directory layout:

  {background_dir}/{train,test}/segments.txt
  {background_dir}/{train,test}/background-{start}-{duration}.hdf5
  {waveforms_dir}/train/{val_waveforms,training_waveforms}.hdf5
  {waveforms_dir}/test/{waveforms,rejected_parameters}.hdf5

The fetch rules wildcard over {start} and {duration}.
Testing waveforms wildcard over {wbranch_id}, which are indices into the
compute_waveform_branches output.
Validation waveforms over {vbranch_id}, a fixed number of jobs that
split num_validation_signals between them.
"""

import json
import math
from pathlib import Path

bg_dir = Path(config["background_dir"])
waveform_dir = Path(config["waveforms_dir"])
train_bg = bg_dir / "train"
test_bg = bg_dir / "test"
train_waveforms = waveform_dir / "train"
test_waveforms = waveform_dir / "test"
data_log_dir = log_dir / "data"

DATA_CONTAINER = container("data")

num_validation_jobs = int(config.get("num_validation_jobs", 200))
validation_branch_ids = [str(i) for i in range(num_validation_jobs)]

num_train_waveform_jobs = int(config.get("num_train_waveform_jobs", 10))
training_branch_ids = [str(i) for i in range(num_train_waveform_jobs)]


localrules:
    compute_waveform_branches,
    # DQSegDB seems to be unreachable from execute points
    generate_train_segments,
    generate_test_segments,


wildcard_constraints:
    start=r"\d{10}",
    duration=r"\d+",
    wbranch_id=r"\d+",
    vbranch_id=r"\d+",
    tbranch_id=r"\d+",


def _fmt_list(values):
    """Format a python list as a jsonargparse CLI value: [a,b]."""
    return "[" + ",".join(str(v) for v in values) + "]"


def _is_analyzeable_segment(start, stop, shifts, psd_length):
    """Whether a segment survives PSD burn-in and the max timeslide."""
    return (stop - start) - max(shifts) - psd_length > 0


def _read_segments(segments_file):
    """Parse a segments file into (start, duration) pairs.

    gwpy writes four whitespace-separated columns per line
    (index, start, stop, duration) after a header line.
    """
    segments = []
    with open(segments_file) as f:
        for line in f:
            parts = line.split()
            try:
                start, duration = float(parts[1]), float(parts[3])
            except ValueError:
                continue
            segments.append((start, duration))
    return segments


def _segment_chunks(segments_file):
    """Split segments into (start, duration) chunks of <= max_duration."""
    max_duration = float(config.get("max_duration", -1))
    chunks = []
    for start, duration in _read_segments(segments_file):
        step = duration if max_duration == -1 else max_duration
        num_steps = (duration - 1) // step + 1
        for i in range(int(num_steps)):
            seg_start = start + i * step
            seg_dur = min(start + duration - seg_start, step)
            chunks.append((int(seg_start), int(seg_dur)))
    return chunks


def get_train_background_files(wildcards):
    """All training background file paths."""
    seg_file = checkpoints.generate_train_segments.get(**wildcards).output[0]
    return [
        str(train_bg / f"background-{s}-{d}.hdf5") for s, d in _segment_chunks(seg_file)
    ]


def get_test_background_files(wildcards):
    """All testing background file paths."""
    seg_file = checkpoints.generate_test_segments.get(**wildcards).output[0]
    return [
        str(test_bg / f"background-{s}-{d}.hdf5") for s, d in _segment_chunks(seg_file)
    ]


def _train_psd_file(wildcards):
    """PSD reference for validation waveforms."""
    return get_train_background_files(wildcards)[-1]


def _test_psd_file(wildcards):
    """PSD reference file for testing waveforms"""
    return get_test_background_files(wildcards)[-1]


def _branch_params(wildcards, input):
    """Read one branch's params from the branch map file."""
    with open(input.branch_map) as f:
        branch = json.load(f)[wildcards.wbranch_id]
    return {
        "start": branch["start"],
        "end": branch["end"],
        "shifts": _fmt_list(branch["shifts"]),
    }


def get_waveform_branch_files(wildcards):
    """Per-branch testing waveform outputs."""
    bmap_file = checkpoints.compute_waveform_branches.get(**wildcards).output[0]
    with open(bmap_file) as f:
        branch_map = json.load(f)
    waveforms = expand(
        str(test_waveforms / "branches" / "{wbranch_id}" / "waveforms.hdf5"),
        wbranch_id=branch_map.keys(),
    )
    rejected = expand(
        str(test_waveforms / "branches" / "{wbranch_id}" / "rejected_parameters.hdf5"),
        wbranch_id=branch_map.keys(),
    )
    return {"waveforms": waveforms, "rejected": rejected}


checkpoint generate_train_segments:
    """Query DQSegDB for valid training data segments."""
    output:
        str(train_bg / "segments.txt"),
    log:
        str(data_log_dir / "generate_train_segments.log"),
    container:
        DATA_CONTAINER
    params:
        flags=_fmt_list(config["flags"]),
        start=config["train_start"],
        end=config["train_end"],
        min_duration=config["train_min_duration"],
        segment_server=config["segment_server"],
    shell:
        "generate-segments"
        " --flags '{params.flags}'"
        " --start {params.start}"
        " --end {params.end}"
        " --min_duration {params.min_duration}"
        " --segment_server {params.segment_server}"
        " --output_file {output}"
        " &> {log}"


checkpoint generate_test_segments:
    """Query DQSegDB for valid test data segments."""
    output:
        str(test_bg / "segments.txt"),
    log:
        str(data_log_dir / "generate_test_segments.log"),
    container:
        DATA_CONTAINER
    params:
        flags=_fmt_list(config["flags"]),
        start=config["test_start"],
        end=config["test_end"],
        min_duration=config["test_min_duration"],
        segment_server=config["segment_server"],
    shell:
        "generate-segments"
        " --flags '{params.flags}'"
        " --start {params.start}"
        " --end {params.end}"
        " --min_duration {params.min_duration}"
        " --segment_server {params.segment_server}"
        " --output_file {output}"
        " &> {log}"


rule fetch_train_background:
    """Download one chunk of training strain data."""
    input:
        str(train_bg / "segments.txt"),
    output:
        str(train_bg / "background-{start}-{duration}.hdf5"),
    log:
        str(data_log_dir / "fetch_train_background-{start}-{duration}.log"),
    container:
        DATA_CONTAINER
    # `fetch` downloads with nproc=3
    threads: 4
    resources:
        **rule_resources("fetch_train_background"),
    params:
        channels=_fmt_list(config["channels"]),
        sample_rate=config["sample_rate"],
        end=lambda wc: int(wc.start) + int(wc.duration),
    shell:
        "fetch-data"
        " --start {wildcards.start}"
        " --end {params.end}"
        " --channels '{params.channels}'"
        " --sample_rate {params.sample_rate}"
        " --output_file {output}"
        " &> {log}"


rule fetch_test_background:
    """Download one chunk of test strain data."""
    input:
        str(test_bg / "segments.txt"),
    output:
        str(test_bg / "background-{start}-{duration}.hdf5"),
    log:
        str(data_log_dir / "fetch_test_background-{start}-{duration}.log"),
    container:
        DATA_CONTAINER
    # `fetch` downloads with nproc=3
    threads: 4
    resources:
        **rule_resources("fetch_test_background"),
    params:
        channels=_fmt_list(config["channels"]),
        sample_rate=config["sample_rate"],
        end=lambda wc: int(wc.start) + int(wc.duration),
    shell:
        "fetch-data"
        " --start {wildcards.start}"
        " --end {params.end}"
        " --channels '{params.channels}'"
        " --sample_rate {params.sample_rate}"
        " --output_file {output}"
        " &> {log}"


checkpoint compute_waveform_branches:
    """Create a file of (start, end, shifts) branches for testing waveforms.

Each branch covers the analyzed part of one test background file at one
timeslide: after the PSD burn-in at the file's start, and before the
timeslide loss (max shift) at its end. Every injection, including its
full waveform, falls inside a single inference branch.

Adds branches until enough data is present to generate as many waveforms
as requested. Loops over background files, adding an additional timeslide
if the target has not yet been met.

Runs locally on the submit node.
"""
    input:
        lambda wildcards: checkpoints.generate_test_segments.get(**wildcards).output[0],
    output:
        str(test_waveforms / "waveform_branch_map.json"),
    run:
        files = [
            (start, start + duration)
            for start, duration in _segment_chunks(input[0])
        ]
        shifts = config["shifts"]
        psd_length = config["psd_length"]
        target = config["num_testing_signals"]
        edge = config["buffer"] + config["waveform_duration"] // 2
        stride = config["spacing"] + config["waveform_duration"]
        branch_map, branch_id, total = {}, 0, 0
        i = 0
        while total < target:
            i += 1
            shift = [i * s for s in shifts]
            added = False
            for start, end in files:
                if total >= target:
                    break
                if not _is_analyzeable_segment(start, end, shift, psd_length):
                    continue
                avail_start = start + psd_length
                avail_end = end - max(shift)
                slots = math.ceil((avail_end - avail_start - 2 * edge) / stride)
                if slots <= 0:
                    continue
                branch_map[str(branch_id)] = {
                    "background": str(
                        test_bg / f"background-{start}-{end - start}.hdf5"
                    ),
                    "start": avail_start,
                    "end": avail_end,
                    "shifts": shift,
                }
                branch_id += 1
                total += slots
                added = True
            # files shrink as the shift grows,
            # stop if nothing is getting added
            if not added:
                break
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        with open(output[0], "w") as f:
            json.dump(branch_map, f, indent=2)


rule testing_waveforms_branch:
    """Generate testing waveforms for one (segment, shifts) branch.

Rejection-samples waveforms against the PSD of the last fetched
test-background chunk and writes the accepted injection set and
the rejected parameters for this branch.
"""
    input:
        branch_map=str(test_waveforms / "waveform_branch_map.json"),
        psd_file=_test_psd_file,
    output:
        waveforms=str(test_waveforms / "branches" / "{wbranch_id}" / "waveforms.hdf5"),
        rejected=str(
            test_waveforms / "branches" / "{wbranch_id}" / "rejected_parameters.hdf5"
        ),
    log:
        str(data_log_dir / "testing_waveforms_branch-{wbranch_id}.log"),
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("testing_waveforms_branch"),
    params:
        branch=_branch_params,
        ifos=_fmt_list(config["ifos"]),
        output_dir=lambda wc, output: str(Path(output.waveforms).parent),
        prior=config["prior"],
        minimum_frequency=config["minimum_frequency"],
        reference_frequency=config["reference_frequency"],
        sample_rate=config["sample_rate"],
        waveform_duration=config["waveform_duration"],
        waveform_approximant=config["waveform_approximant"],
        right_pad=config["right_pad"],
        highpass=config["highpass"],
        lowpass=config["lowpass"] or "null",
        snr_threshold=config["snr_threshold"],
        spacing=config["spacing"],
        buffer=config["buffer"],
        max_num_samples=config["max_num_samples"],
        seed=config["seed"],
    shell:
        "generate-testing-waveforms"
        " --start {params.branch[start]}"
        " --end {params.branch[end]}"
        " --ifos '{params.ifos}'"
        " --shifts '{params.branch[shifts]}'"
        " --spacing {params.spacing}"
        " --buffer {params.buffer}"
        " --prior {params.prior}"
        " --minimum_frequency {params.minimum_frequency}"
        " --reference_frequency {params.reference_frequency}"
        " --sample_rate {params.sample_rate}"
        " --waveform_duration {params.waveform_duration}"
        " --waveform_approximant {params.waveform_approximant}"
        " --right_pad {params.right_pad}"
        " --highpass {params.highpass}"
        " --lowpass {params.lowpass}"
        " --snr_threshold {params.snr_threshold}"
        " --psd_file {input.psd_file}"
        " --max_num_samples {params.max_num_samples}"
        " --seed {params.seed}"
        " --output_dir {params.output_dir}"
        " &> {log}"


rule aggregate_testing_waveforms:
    """Merge per-branch testing waveforms into the final injection set."""
    input:
        unpack(get_waveform_branch_files),
    output:
        waveforms=str(test_waveforms / "waveforms.hdf5"),
        rejected=str(test_waveforms / "rejected_parameters.hdf5"),
    log:
        str(data_log_dir / "aggregate_testing_waveforms.log"),
    localrule: config.get("aggregate_rules_local", False)
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("aggregate_testing_waveforms"),
    params:
        ifos=config["ifos"],
    script:
        "scripts/aggregate_testing_waveforms.py"


rule val_waveforms_branch:
    """Generate one branch of validation waveforms via rejection sampling.

num_validation_signals is split evenly across num_validation_jobs branches.
The PSD reference is the last fetched train-background chunk.
"""
    input:
        psd_file=_train_psd_file,
    output:
        str(train_waveforms / "validation_tmp" / "waveforms-{vbranch_id}.hdf5"),
    log:
        str(data_log_dir / "val_waveforms_branch-{vbranch_id}.log"),
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("val_waveforms_branch"),
    params:
        num_signals=math.ceil(config["num_validation_signals"] / num_validation_jobs),
        ifos=_fmt_list(config["ifos"]),
        prior=config["prior"],
        minimum_frequency=config["minimum_frequency"],
        reference_frequency=config["reference_frequency"],
        sample_rate=config["sample_rate"],
        waveform_duration=config["waveform_duration"],
        waveform_approximant=config["waveform_approximant"],
        right_pad=config["right_pad"],
        highpass=config["highpass"],
        lowpass=config["lowpass"] or "null",
        snr_threshold=config["snr_threshold"],
        max_num_samples=config["max_num_samples"],
    shell:
        "generate-validation-waveforms"
        " --num_signals {params.num_signals}"
        " --ifos '{params.ifos}'"
        " --prior {params.prior}"
        " --minimum_frequency {params.minimum_frequency}"
        " --reference_frequency {params.reference_frequency}"
        " --sample_rate {params.sample_rate}"
        " --waveform_duration {params.waveform_duration}"
        " --waveform_approximant {params.waveform_approximant}"
        " --right_pad {params.right_pad}"
        " --highpass {params.highpass}"
        " --lowpass {params.lowpass}"
        " --snr_threshold {params.snr_threshold}"
        " --psd {input.psd_file}"
        " --max_num_samples {params.max_num_samples}"
        " --output_file {output}"
        " &> {log}"


rule aggregate_val_waveforms:
    """Merge per-branch validation waveforms into val_waveforms.hdf5."""
    input:
        expand(
            str(train_waveforms / "validation_tmp" / "waveforms-{vbranch_id}.hdf5"),
            vbranch_id=validation_branch_ids,
        ),
    output:
        str(train_waveforms / "val_waveforms.hdf5"),
    log:
        str(data_log_dir / "aggregate_val_waveforms.log"),
    localrule: config.get("aggregate_rules_local", False)
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("aggregate_val_waveforms"),
    params:
        ifos=config["ifos"],
    script:
        "scripts/aggregate_val_waveforms.py"


if config.get("pregenerate_training_waveforms", False):

    rule training_waveforms_branch:
        """Generate one branch of training waveform polarizations.

        num_training_signals is split evenly across num_train_waveform_jobs branches.
        """
        output:
            str(train_waveforms / "training_tmp" / "{tbranch_id}.hdf5"),
        log:
            str(data_log_dir / "training_waveforms_branch-{tbranch_id}.log"),
        container:
            DATA_CONTAINER
        resources:
            **rule_resources("training_waveforms_branch"),
        params:
            num_signals=math.ceil(
                config["num_training_signals"] / num_train_waveform_jobs
            ),
            sample_rate=config["sample_rate"],
            waveform_duration=config["waveform_duration"],
            prior=config["prior"],
            minimum_frequency=config["minimum_frequency"],
            reference_frequency=config["reference_frequency"],
            waveform_approximant=config["waveform_approximant"],
            right_pad=config["right_pad"],
        shell:
            "generate-training-waveforms"
            " --num_signals {params.num_signals}"
            " --sample_rate {params.sample_rate}"
            " --waveform_duration {params.waveform_duration}"
            " --prior {params.prior}"
            " --minimum_frequency {params.minimum_frequency}"
            " --reference_frequency {params.reference_frequency}"
            " --waveform_approximant {params.waveform_approximant}"
            " --right_pad {params.right_pad}"
            " --output_file {output}"
            " &> {log}"

    rule aggregate_training_waveforms:
        """Merge per-branch training waveforms into training_waveforms.hdf5."""
        input:
            expand(
                str(train_waveforms / "training_tmp" / "{tbranch_id}.hdf5"),
                tbranch_id=training_branch_ids,
            ),
        output:
            str(train_waveforms / "training_waveforms.hdf5"),
        log:
            str(data_log_dir / "aggregate_training_waveforms.log"),
        localrule: config.get("aggregate_rules_local", False)
        container:
            DATA_CONTAINER
        resources:
            **rule_resources("aggregate_training_waveforms"),
        script:
            "scripts/aggregate_training_waveforms.py"
