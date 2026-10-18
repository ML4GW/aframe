"""Snakemake rules for data acquisition and waveform generation.

Rules:
  generate_segments             checkpoint: query DQSegDB for a split's segments
  fetch_background              download one chunk of a split's strain data
  compute_psd                   PSDs that waveform jobs rejection-sample against
  val_waveforms_branch          validation waveforms for one branch
  aggregate_val_waveforms       merge per-branch validation waveforms
  val_waveforms                 generate validation waveforms
  testing_waveforms_branch      testing waveforms for one branch
  aggregate_testing_waveforms   merge per-branch testing waveforms
  testing_waveforms             generate testing waveforms
  training_waveforms_branch     one branch of training waveform polarizations (optional)
  aggregate_training_waveforms  merge per-branch training waveforms (optional)
  training_waveforms            generate training waveforms

Directory layout:

  data/{train,test}/segments.txt
  waveforms/train/{population}/psd.hdf5
  waveforms/train/training_waveforms/training_waveforms_{population}.hdf5
  waveforms/train/val_waveforms/val_waveforms_{population}.hdf5
  waveforms/test/{population}/{waveforms,rejected_parameters,psd}.hdf5

Every waveform is its population's waveform_duration long.
Segments, fetching and PSDs wildcard over {split}, "train" or "test", and
fetching over each file's {start} and {duration}.
Every waveform rule wildcards over a {population} (see Populations below).
Testing waveforms wildcard over a test file's {start} and {duration} and
a {timeslide}, from testing_waveform_branches.
Validation and training waveforms over {vbranch_id} and {tbranch_id}, the
population's num_validation_jobs and num_train_waveform_jobs jobs, which
split its num_validation_signals and num_training_signals between them.
"""

import math
import re
from pathlib import Path

bg_dir = run_dir / "data"
waveform_dir = run_dir / "waveforms"
train_bg = bg_dir / "train"
test_bg = bg_dir / "test"
train_waveforms = waveform_dir / "train"
test_waveforms = waveform_dir / "test"
training_waveforms_dir = train_waveforms / "training_waveforms"
val_waveforms_dir = train_waveforms / "val_waveforms"
training_waveforms_file = str(
    training_waveforms_dir / "training_waveforms_{population}.hdf5"
)
val_waveforms_file = str(val_waveforms_dir / "val_waveforms_{population}.hdf5")
data_log_dir = log_dir / "data"

DATA_CONTAINER = container("data")


# See `populations` in pipeline/config/config.yaml. These are the keys a
# population can set, which exist only inside a population. It only needs
# the ones of the tasks it's used for, which its jobs read through pcfg():
#   training:          prior, waveform_approximant, minimum_frequency,
#                      reference_frequency, waveform_duration,
#                      num_training_signals, num_train_waveform_jobs,
#                      pregenerate_training_waveforms
#   validation:        prior, waveform_approximant, minimum_frequency,
#                      reference_frequency, waveform_duration, snr_threshold,
#                      num_validation_signals, num_validation_jobs
#   testing:           prior, waveform_approximant, minimum_frequency,
#                      reference_frequency, waveform_duration, snr_threshold,
#                      max_num_samples, num_testing_signals, spacing, buffer,
#   sensitive volume:  source_prior, mass_combos


POPULATIONS = config["active_populations"]
if isinstance(POPULATIONS, str):
    POPULATIONS = [POPULATIONS]


def pcfg(population, key):
    """A population's value of `key`, which is reported missing here, by the
    first job of this run that needs it.
    """
    try:
        return config["populations"][population][key]
    except KeyError:
        raise WorkflowError(
            f"Population {population!r} doesn't set {key!r}, which this run "
            f"needs. Add it under `populations: {population}:`."
        ) from None


# Training has one waveform_sampler (in train_config), which either loads
# every population's training waveforms from disk or generates them all on
# the fly, so every active population must agree on this.
_pregenerate = {
    pcfg(p, "pregenerate_training_waveforms")
    for p in POPULATIONS
    if "pregenerate_training_waveforms" in config["populations"][p]
}
if len(_pregenerate) > 1:
    raise WorkflowError(
        "Active populations must all set the same "
        "pregenerate_training_waveforms, since training either loads every "
        "population's training waveforms from disk or generates them all."
    )
PREGENERATE_TRAINING_WAVEFORMS = _pregenerate.pop() if _pregenerate else None


def _population_params(*keys):
    """Rule params that read each of `keys` for the job's {population}."""
    return {key: lambda wc, key=key: pcfg(wc.population, key) for key in keys}


def branch_ids(population, jobs_key):
    """IDs of a population's validation or training waveform branches."""
    return [str(i) for i in range(pcfg(population, jobs_key))]


def validation_seed(population, vbranch_id):
    """A validation branch's seed, distinct across every population's."""
    earlier = POPULATIONS[: POPULATIONS.index(population)]
    offset = sum(pcfg(p, "num_validation_jobs") for p in earlier)
    return config["seed"] + offset + int(vbranch_id)


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
    population="|".join(POPULATIONS),


def _fmt_list(values):
    """Format a python list as a jsonargparse CLI value: [a,b]."""
    return "[" + ",".join(str(v) for v in values) + "]"


def _is_analyzeable_segment(start, stop, shifts, psd_length):
    """Whether a segment survives PSD burn-in and the max timeslide."""
    return (stop - start) - max(shifts) - psd_length > 0


def timeslide_shifts(timeslide):
    """Each detector's time shift, in seconds, at a timeslide.

    Timeslide n shifts each detector by n of its `shift_steps`. Timeslide 0
    is zero lag.
    """
    return [int(timeslide) * step for step in config["shift_steps"]]


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


def test_background_files(wildcards):
    """Every test background file, for the test_background target.

    The list of files isn't known until generate_segments has queried the
    test segments.

    Snakemake passes every such function the rule's wildcards. They aren't
    used here, but the argument is required.
    """
    return background_files("test")


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
    """The (start, duration, timeslide, population) of each testing branch.

    A branch is one test background file, identified by its start and
    duration, analyzed at one timeslide (see timeslide_shifts). These are
    the same (file, timeslide) pairs that inference runs on, so every
    injection falls inside a single inference branch.

    Each population, in active_populations order, gets the next block of
    timeslides, adding branches until they have room for its
    num_testing_signals injections. The next population starts at the
    following timeslide, so a timeslide never mixes populations.

    Branches are named by population, file and timeslide rather than
    numbered, so their files stay valid in a waveforms_dir shared between runs.
    """
    seg_file = checkpoints.generate_segments.get(split="test").output[0]
    chunks = _segment_chunks(seg_file)
    psd_length = config["psd_length"]
    branches, timeslide = [], 0
    for population in POPULATIONS:
        target, total = pcfg(population, "num_testing_signals"), 0
        waveform_duration = pcfg(population, "waveform_duration")
        edge = pcfg(population, "buffer") + waveform_duration // 2
        stride = pcfg(population, "spacing") + waveform_duration
        while total < target:
            timeslide += 1
            shifts = timeslide_shifts(timeslide)
            added = False
            for start, duration in chunks:
                if total >= target:
                    break
                if not _is_analyzeable_segment(
                    start, start + duration, shifts, psd_length
                ):
                    continue
                span = duration - psd_length - max(shifts)
                slots = math.ceil((span - 2 * edge) / stride)
                if slots <= 0:
                    continue
                branches.append((start, duration, timeslide, population))
                total += slots
                added = True
            # the analyzed part of each file shrinks as the shifts grow,
            # stop if nothing is getting added
            if not added:
                break
        if not total:
            raise WorkflowError(
                f"No room for {population} injections from timeslide "
                f"{timeslide} on. Lower an earlier population's "
                "num_testing_signals, or this one's waveform_duration."
            )
    return branches


testing_branch_dir = (
    test_waveforms / "{population}" / "branches" / "{start}-{duration}-{timeslide}"
)


def testing_waveform_file(start, duration, timeslide, population, name="waveforms"):
    """One testing waveform branch's output file."""
    return str(testing_branch_dir / f"{name}.hdf5").format(
        start=start, duration=duration, timeslide=timeslide, population=population
    )


def get_testing_waveform_files(wildcards):
    """A population's testing waveform branch outputs, keyed by output name."""
    branches = [b for b in testing_waveform_branches() if b[3] == wildcards.population]
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
    # Proprietary frames (full channel names) are only readable on the site's
    # own nodes, not on glideins. Open data (bare IFO names) can come from
    # anywhere.
    resources:
        **rule_resources(
            "fetch_background",
            "data",
            local_pool=any(":" in channel for channel in config["channels"]),
        ),
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


rule train_background:
    """Target rule: fetch every training background file.

Runs no command. Its inputs make snakemake query the training segments,
then run fetch_background for each chunk.
"""
    input:
        train_background_files,
    localrule: True


rule test_background:
    """Target rule: fetch every testing background file.

Runs no command. Its inputs make snakemake query the training segments,
then run fetch_background for each chunk.
"""
    input:
        test_background_files,
    localrule: True


rule compute_psd:
    """Compute the PSDs that a population's waveform jobs in a split
rejection-sample against so that each job reads a small file instead of
a full background file.
"""
    input:
        _psd_background_file,
    output:
        str(waveform_dir / "{split}" / "{population}" / "psd.hdf5"),
    log:
        str(data_log_dir / "compute_psd-{split}-{population}.log"),
    container:
        DATA_CONTAINER
    params:
        ifos=config["ifos"],
        df=lambda wc: 1 / pcfg(wc.population, "waveform_duration"),
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
        psd_file=str(test_waveforms / "{population}" / "psd.hdf5"),
    output:
        waveforms=str(testing_branch_dir / "waveforms.hdf5"),
        rejected=str(testing_branch_dir / "rejected_parameters.hdf5"),
    log:
        str(
            data_log_dir
            / "testing_waveforms_branch-{population}-{start}-{duration}-{timeslide}.log"
        ),
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("testing_waveforms_branch", "data"),
    params:
        **_population_params(
            "prior",
            "minimum_frequency",
            "reference_frequency",
            "waveform_duration",
            "waveform_approximant",
            "snr_threshold",
            "spacing",
            "buffer",
            "max_num_samples",
        ),
        start=lambda wc: int(wc.start) + config["psd_length"],
        end=lambda wc: (
            int(wc.start) + int(wc.duration) - max(timeslide_shifts(wc.timeslide))
        ),
        shifts=lambda wc: _fmt_list(timeslide_shifts(wc.timeslide)),
        ifos=_fmt_list(config["ifos"]),
        sample_rate=config["sample_rate"],
        right_pad=config["right_pad"],
        highpass=config["highpass"],
        lowpass=config["lowpass"] or "null",
        jitter=config["jitter"],
        seed=config["seed"],
    shell:
        "generate-testing-waveforms"
        " --start {params.start}"
        " --end {params.end}"
        " --ifos '{params.ifos}'"
        " --shifts '{params.shifts}'"
        " --spacing {params.spacing}"
        " --buffer {params.buffer}"
        " --jitter {params.jitter}"
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
    """Merge a population's per-branch testing waveforms into its injection set."""
    input:
        unpack(get_testing_waveform_files),
    output:
        waveforms=str(test_waveforms / "{population}" / "waveforms.hdf5"),
        rejected_parameters=str(
            test_waveforms / "{population}" / "rejected_parameters.hdf5"
        ),
    log:
        str(data_log_dir / "aggregate_testing_waveforms-{population}.log"),
    localrule: config["aggregate_rules_local"]
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("aggregate_testing_waveforms", "data"),
    params:
        ifos=config["ifos"],
        classes={"waveforms": "responses", "rejected_parameters": "parameters"},
    script:
        "scripts/aggregate_waveforms.py"


rule val_waveforms_branch:
    """Generate one branch of a population's validation waveforms via
rejection sampling

num_validation_signals is split evenly across num_validation_jobs branches.
The PSDs are those of the last fetched train-background chunk.
"""
    input:
        psd_file=str(train_waveforms / "{population}" / "psd.hdf5"),
    output:
        str(val_waveforms_dir / "{population}" / "waveforms-{vbranch_id}.hdf5"),
    log:
        str(data_log_dir / "val_waveforms_branch-{population}-{vbranch_id}.log"),
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("val_waveforms_branch", "data"),
    params:
        **_population_params(
            "prior",
            "minimum_frequency",
            "reference_frequency",
            "waveform_duration",
            "waveform_approximant",
            "snr_threshold",
            "max_num_samples",
        ),
        num_signals=lambda wc: math.ceil(
            pcfg(wc.population, "num_validation_signals")
            / pcfg(wc.population, "num_validation_jobs")
        ),
        ifos=_fmt_list(config["ifos"]),
        sample_rate=config["sample_rate"],
        right_pad=config["right_pad"],
        highpass=config["highpass"],
        lowpass=config["lowpass"] or "null",
        # offset by branch so each branch samples different parameters
        seed=lambda wc: validation_seed(wc.population, wc.vbranch_id),
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
        " --seed {params.seed}"
        " --output_file {output}"
        " &> {log}"


rule aggregate_val_waveforms:
    """Merge population's per-branch validation waveforms into val_waveforms/val_waveforms_{population}.hdf5"""
    input:
        waveforms=lambda wc: expand(
            str(val_waveforms_dir / "{population}" / "waveforms-{vbranch_id}.hdf5"),
            population=wc.population,
            vbranch_id=branch_ids(wc.population, "num_validation_jobs"),
        ),
    output:
        waveforms=str(val_waveforms_dir / "val_waveforms_{population}.hdf5"),
    log:
        str(data_log_dir / "aggregate_val_waveforms-{population}.log"),
    localrule: config["aggregate_rules_local"]
    container:
        DATA_CONTAINER
    resources:
        **rule_resources("aggregate_val_waveforms", "data"),
    params:
        ifos=config["ifos"],
        classes={"waveforms": "waveforms"},
    script:
        "scripts/aggregate_waveforms.py"


rule testing_waveforms:
    """Target rule: every active population's testing waveforms.

Runs no command. The aggregate rules have a {population} wildcard, so
they can't be run by name; this asks for all of their outputs instead.
"""
    input:
        expand(
            str(test_waveforms / "{population}" / "waveforms.hdf5"),
            population=POPULATIONS,
        ),
    localrule: True


rule val_waveforms:
    """Target rule: every active population's validation waveforms."""
    input:
        expand(
            str(val_waveforms_dir / "val_waveforms_{population}.hdf5"),
            population=POPULATIONS,
        ),
    localrule: True


if PREGENERATE_TRAINING_WAVEFORMS:

    rule training_waveforms_branch:
        """Generate one branch of a population's training polarizations.

        num_training_signals is split evenly across num_train_waveform_jobs branches.
        """
        output:
            str(training_waveforms_dir / "{population}" / "{tbranch_id}.hdf5"),
        log:
            str(
                data_log_dir
                / "training_waveforms_branch-{population}-{tbranch_id}.log"
            ),
        container:
            DATA_CONTAINER
        resources:
            **rule_resources("training_waveforms_branch", "data"),
        params:
            **_population_params(
                "prior",
                "minimum_frequency",
                "reference_frequency",
                "waveform_duration",
                "waveform_approximant",
            ),
            num_signals=lambda wc: math.ceil(
                pcfg(wc.population, "num_training_signals")
                / pcfg(wc.population, "num_train_waveform_jobs")
            ),
            sample_rate=config["sample_rate"],
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

    rule training_waveforms:
        """Target rule: every active population's training waveforms."""
        input:
            expand(
                str(training_waveforms_dir / "training_waveforms_{population}.hdf5"),
                population=POPULATIONS,
            ),
        localrule: True

    rule aggregate_training_waveforms:
        """Merge a population's per-branch training waveforms into training_waveforms_{population}.hdf5."""
        input:
            waveforms=lambda wc: expand(
                str(training_waveforms_dir / "{population}" / "{tbranch_id}.hdf5"),
                population=wc.population,
                tbranch_id=branch_ids(wc.population, "num_train_waveform_jobs"),
            ),
        output:
            waveforms=str(
                training_waveforms_dir / "training_waveforms_{population}.hdf5"
            ),
        log:
            str(data_log_dir / "aggregate_training_waveforms-{population}.log"),
        localrule: config["aggregate_rules_local"]
        container:
            DATA_CONTAINER
        resources:
            **rule_resources("aggregate_training_waveforms", "data"),
        params:
            ifos=config["ifos"],
            classes={"waveforms": "polarizations"},
        script:
            "scripts/aggregate_waveforms.py"
