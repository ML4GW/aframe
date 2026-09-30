"""Snakemake rules for data acquisition and waveform generation.

Rules:
  generate_segments             checkpoint: query DQSegDB for a split's segments
  fetch_background              download one chunk of a split's strain data
  compute_psd                   PSDs that waveform jobs rejection-sample against
  val_waveforms_branch          validation waveforms for one branch
  aggregate_val_waveforms       merge per-branch validation waveforms
  testing_waveforms_branch      testing waveforms for one branch
  aggregate_testing_waveforms   merge per-branch testing waveforms
  training_waveforms_branch     one branch of training waveform polarizations (optional)
  aggregate_training_waveforms  merge per-branch training waveforms (optional)

Directory layout:

  data/{train,test}/segments.txt
  data/{train,test}/background-{start}-{duration}.hdf5
  waveforms/train/{val_waveforms,training_waveforms,psd}.hdf5
  waveforms/test/{waveforms,rejected_parameters,psd}.hdf5

Segments, fetching and PSDs wildcard over {split}, "train" or "test", and
fetching over each file's {start} and {duration}.
Testing waveforms wildcard over a test file's {start} and {duration} and
a {timeslide}, from testing_waveform_branches.
Validation waveforms over {vbranch_id}, a fixed number of jobs that
split num_validation_signals between them.
"""

import math
from pathlib import Path

bg_dir = run_dir / "data"
waveform_dir = run_dir / "waveforms"
train_bg = bg_dir / "train"
test_bg = bg_dir / "test"
train_waveforms = waveform_dir / "train"
test_waveforms = waveform_dir / "test"
data_log_dir = log_dir / "data"

DATA_CONTAINER = container("data")

num_validation_jobs = int(config["num_validation_jobs"])
validation_branch_ids = [str(i) for i in range(num_validation_jobs)]

num_train_waveform_jobs = int(config["num_train_waveform_jobs"])
training_branch_ids = [str(i) for i in range(num_train_waveform_jobs)]


localrules:
    compute_psd,
    # DQSegDB seems to be unreachable from execute points
    generate_segments,


wildcard_constraints:
    split="train|test",
    start=r"\d{10}",
    duration=r"\d+",
    timeslide=r"\d+",
    vbranch_id=r"\d+",
    tbranch_id=r"\d+",


def _fmt_list(values):
    """Format a python list as a jsonargparse CLI value: [a,b]."""
    return "[" + ",".join(str(v) for v in values) + "]"


def _is_analyzeable_segment(start, stop, shifts, psd_length):
    """Whether a segment survives PSD burn-in and the max timeslide."""
    return (stop - start) - max(shifts) - psd_length > 0


def timeslide_shifts(timeslide):
    """Each detector's time shift, in seconds, at a timeslide.

    `shifts` in the config is the step each detector moves per timeslide,
    so timeslide n shifts each detector by n of its steps.
    Timeslide 0 is zero lag.
    """
    return [int(timeslide) * step for step in config["shifts"]]


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
    max_duration = float(config["max_duration"])
    chunks = []
    for start, duration in _read_segments(segments_file):
        step = duration if max_duration == -1 else max_duration
        num_steps = (duration - 1) // step + 1
        for i in range(int(num_steps)):
            seg_start = start + i * step
            seg_dur = min(start + duration - seg_start, step)
            chunks.append((int(seg_start), int(seg_dur)))
    return chunks


def background_files(split):
    """All background file paths of a split, "train" or "test"."""
    seg_file = checkpoints.generate_segments.get(split=split).output[0]
    return [
        str(bg_dir / split / f"background-{s}-{d}.hdf5")
        for s, d in _segment_chunks(seg_file)
    ]


def train_background_files(wildcards):
    """Every training background file as an input for the training rules.

    The list of files isn't known until generate_segments has queried the
    training segments, which happens partway through a run. Because a rule
    can't list all the files directly, it names this function as its input,
    list these files directly. Instead it names this function as its input,
    `background=train_background_files`, without calling it, and snakemake
    calls it later.

    Snakemake passes every such function the rule's wildcards. They aren't
    used here, but the argument is required.
    """
    return background_files("train")


# Shortest background file to compute waveform PSDs from
PSD_MIN_DURATION = 2048


def _psd_background_file(wildcards):
    """
    The background file from which to compute PSDs that waveforms are
    rejection-sampled against. We take the last file of the split that
    is at least PSD_MIN_DURATION long.
    """
    for fname in reversed(background_files(wildcards.split)):
        if int(Path(fname).stem.split("-")[-1]) >= PSD_MIN_DURATION:
            return fname
    raise WorkflowError(
        f"No {wildcards.split} background file is at least "
        f"{PSD_MIN_DURATION} s long"
    )


def testing_waveform_branches():
    """The (start, duration, timeslide) of each testing waveform branch.

    A branch is one test background file, identified by its start and
    duration, analyzed at one timeslide (see timeslide_shifts). These are
    the same (file, timeslide) pairs that inference runs on, so every
    injection falls inside a single inference branch.

    Branches are added until they have room for num_testing_signals injections.

    Branches are named by file and timeslide rather than numbered, so their
    files stay valid in a waveforms_dir shared between runs.
    """
    seg_file = checkpoints.generate_segments.get(split="test").output[0]
    chunks = _segment_chunks(seg_file)
    psd_length = config["psd_length"]
    target = config["num_testing_signals"]
    edge = config["buffer"] + config["waveform_duration"] // 2
    stride = config["spacing"] + config["waveform_duration"]
    branches, total, timeslide = [], 0, 0
    while total < target:
        timeslide += 1
        shifts = timeslide_shifts(timeslide)
        added = False
        for start, duration in chunks:
            if total >= target:
                break
            if not _is_analyzeable_segment(start, start + duration, shifts, psd_length):
                continue
            span = duration - psd_length - max(shifts)
            slots = math.ceil((span - 2 * edge) / stride)
            if slots <= 0:
                continue
            branches.append((start, duration, timeslide))
            total += slots
            added = True
        # the analyzed part of each file shrinks as the shifts grow,
        # stop if nothing is getting added
        if not added:
            break
    return branches


testing_branch_dir = test_waveforms / "branches" / "{start}-{duration}-{timeslide}"


def testing_waveform_file(start, duration, timeslide, name="waveforms"):
    """One testing waveform branch's output file."""
    return str(testing_branch_dir / f"{name}.hdf5").format(
        start=start, duration=duration, timeslide=timeslide
    )


def get_testing_waveform_files(wildcards):
    """Every testing waveform branch's outputs, keyed by output name."""
    branches = testing_waveform_branches()
    return {
        name: [testing_waveform_file(*b, name) for b in branches]
        for name in ["waveforms", "rejected_parameters"]
    }


checkpoint generate_segments:
    """Query DQSegDB for a split's valid data segments."""
    output:
        str(bg_dir / "{split}" / "segments.txt"),
    log:
        str(data_log_dir / "generate_segments-{split}.log"),
    container:
        DATA_CONTAINER
    params:
        flags=_fmt_list(config["flags"]),
        start=lambda wc: config[f"{wc.split}_start"],
        end=lambda wc: config[f"{wc.split}_end"],
        min_duration=lambda wc: config[f"{wc.split}_min_duration"],
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


rule fetch_background:
    """Download one chunk of a split's strain data."""
    input:
        str(bg_dir / "{split}" / "segments.txt"),
    output:
        str(bg_dir / "{split}" / "background-{start}-{duration}.hdf5"),
    log:
        str(data_log_dir / "fetch_background-{split}-{start}-{duration}.log"),
    container:
        DATA_CONTAINER
    # `fetch` downloads with nproc=3
    threads: 4
    resources:
        **rule_resources("fetch_background"),
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


rule compute_psd:
    """Compute the PSDs that a split's waveform jobs rejection-sample
against so that each job reads a small file instead of a full
background file.
"""
    input:
        _psd_background_file,
    output:
        str(waveform_dir / "{split}" / "psd.hdf5"),
    log:
        str(data_log_dir / "compute_psd-{split}.log"),
    container:
        DATA_CONTAINER
    params:
        ifos=config["ifos"],
        df=1 / config["waveform_duration"],
    script:
        "scripts/compute_psd.py"


rule testing_waveforms_branch:
    """Generate testing waveforms for one branch: the analyzed part of
the test background file starting at {start}, lasting {duration}, at
timeslide {timeslide}. The PSD burn-in at the file's start and
the timeslide loss (max shift) at its end are accounted for.

Rejection-samples waveforms against the PSDs of the last fetched
test-background chunk and writes the accepted injection set and
the rejected parameters for this branch.
"""
    input:
        psd_file=str(test_waveforms / "psd.hdf5"),
    output:
        waveforms=str(testing_branch_dir / "waveforms.hdf5"),
        rejected=str(testing_branch_dir / "rejected_parameters.hdf5"),
    log:
        str(
            data_log_dir
            / "testing_waveforms_branch-{start}-{duration}-{timeslide}.log"
        ),
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("testing_waveforms_branch"),
    params:
        start=lambda wc: int(wc.start) + config["psd_length"],
        end=lambda wc: (
            int(wc.start) + int(wc.duration) - max(timeslide_shifts(wc.timeslide))
        ),
        shifts=lambda wc: _fmt_list(timeslide_shifts(wc.timeslide)),
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
        spacing=config["spacing"],
        buffer=config["buffer"],
        max_num_samples=config["max_num_samples"],
        seed=config["seed"],
    shell:
        "generate-testing-waveforms"
        " --start {params.start}"
        " --end {params.end}"
        " --ifos '{params.ifos}'"
        " --shifts '{params.shifts}'"
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
        " --waveforms_file {output.waveforms}"
        " --rejected_file {output.rejected}"
        " &> {log}"


rule aggregate_testing_waveforms:
    """Merge per-branch testing waveforms into the final injection set."""
    input:
        unpack(get_testing_waveform_files),
    output:
        waveforms=str(test_waveforms / "waveforms.hdf5"),
        rejected_parameters=str(test_waveforms / "rejected_parameters.hdf5"),
    log:
        str(data_log_dir / "aggregate_testing_waveforms.log"),
    localrule: config["aggregate_rules_local"]
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("aggregate_testing_waveforms"),
    params:
        ifos=config["ifos"],
        classes={"waveforms": "responses", "rejected_parameters": "parameters"},
    script:
        "scripts/aggregate_waveforms.py"


rule val_waveforms_branch:
    """Generate one branch of validation waveforms via rejection sampling.

num_validation_signals is split evenly across num_validation_jobs branches.
The PSDs are those of the last fetched train-background chunk.
"""
    input:
        psd_file=str(train_waveforms / "psd.hdf5"),
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
        waveforms=expand(
            str(train_waveforms / "validation_tmp" / "waveforms-{vbranch_id}.hdf5"),
            vbranch_id=validation_branch_ids,
        ),
    output:
        waveforms=str(train_waveforms / "val_waveforms.hdf5"),
    log:
        str(data_log_dir / "aggregate_val_waveforms.log"),
    localrule: config["aggregate_rules_local"]
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("aggregate_val_waveforms"),
    params:
        ifos=config["ifos"],
        classes={"waveforms": "waveforms"},
    script:
        "scripts/aggregate_waveforms.py"


if config["pregenerate_training_waveforms"]:

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
            waveforms=expand(
                str(train_waveforms / "training_tmp" / "{tbranch_id}.hdf5"),
                tbranch_id=training_branch_ids,
            ),
        output:
            waveforms=str(train_waveforms / "training_waveforms.hdf5"),
        log:
            str(data_log_dir / "aggregate_training_waveforms.log"),
        localrule: config["aggregate_rules_local"]
        container:
            DATA_CONTAINER
        resources:
            **rule_resources("aggregate_training_waveforms"),
        params:
            ifos=config["ifos"],
            classes={"waveforms": "polarizations"},
        script:
            "scripts/aggregate_waveforms.py"
