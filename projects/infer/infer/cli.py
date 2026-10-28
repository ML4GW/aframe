"""Inference CLI for groups of branches.

Two entry points:
    infer-local: run branches in-process on a GPU, several shifts of a file
        at a time, through main.infer_local().
    infer-triton: stream branches one at a time to a running Triton server,
        through main.infer_triton().

Both use the same Postprocessor and write the same outputs, so
aggregate_infer is client agnostic.

--analysis_type selects the data: hdf5 (timeslide background + injections)
or rnp (Rates and Populations frames).
"""

import json
import logging
import tempfile
import time
from pathlib import Path

import h5py
import jsonargparse
from ledger.events import EventSet, RecoveredInjectionSet
from utils.logging import configure_logging

from infer.aggregate import merge_timeseries, write_timeseries
from infer.data import RnPStreamSequence, ShiftBatchSequence
from infer.main import infer_local, infer_triton
from infer.postprocess import Postprocessor


def _write_outputs(outdir, branch_id, results, seq, postproc, cfg):
    background, foreground, background_ts, foreground_ts = results
    outdir.mkdir(parents=True, exist_ok=True)
    background.write(outdir / "background.hdf5")
    foreground.write(outdir / "foreground.hdf5")
    if cfg.timeseries_out is not None:
        write_timeseries(
            outdir / "timeseries.hdf5",
            branch_id,
            background_ts,
            foreground_ts,
            t0=seq.t0,
            sample_t0=postproc.stream_outputs.availability_time(seq.t0, 0),
            inference_sampling_rate=cfg.inference_sampling_rate,
            shifts=postproc.shifts,
        )


def _postprocessor(cfg, t0, shifts) -> Postprocessor:
    return Postprocessor(
        t0=t0,
        shifts=shifts,
        psd_length=cfg.psd_length,
        fduration=cfg.fduration,
        inference_sampling_rate=cfg.inference_sampling_rate,
        integration_window_length=cfg.integration_window_length,
        cluster_window_length=cfg.cluster_window_length,
    )


def _sequence(cfg, batch):
    """The stream sequence for a batch of branches that share one file."""
    fname = batch[0][1]["fname"]
    if cfg.analysis_type == "rnp":
        return RnPStreamSequence(
            Path(fname),
            cfg.channel,
            cfg.ifos,
            cfg.sample_rate,
            cfg.inference_sampling_rate,
            cfg.batch_size,
        )
    return ShiftBatchSequence(
        fname,
        [branch["waveforms"] for _, branch in batch],
        cfg.ifos,
        [branch["shifts"] for _, branch in batch],
        cfg.inference_sampling_rate,
        cfg.batch_size,
    )


def _run_triton(client, cfg, branches: dict, scratch: Path) -> None:
    """Stream a group's branches to the Triton server one at a time."""
    with client:
        for branch_id, branch in branches.items():
            sequence = _sequence(cfg, [(branch_id, branch)])
            postprocessor = _postprocessor(cfg, sequence.t0, branch["shifts"])

            # Branch IDs are unique within a run and a branch has at most two
            # streams, so 2 * ID + 1 gives every stream a unique sequence ID.
            # Triton handles an ID of 0 as a special case, so avoid it.
            base_id = 2 * int(branch_id) + 1
            results = infer_triton(
                client,
                sequence,
                [postprocessor],
                base_id,
                cfg.rate,
            )
            _write_outputs(
                scratch / branch_id,
                branch_id,
                results[0],
                sequence,
                postprocessor,
                cfg,
            )


def _run_local(client, cfg, branches: dict, scratch: Path) -> None:
    """Run a group's branches in-process, in batches of shifts_per_batch
    shifts. Each R&P branch is its own frame, so R&P runs use a
    shifts_per_batch of 1."""
    items = list(branches.items())
    batches = [
        items[i : i + cfg.shifts_per_batch]
        for i in range(0, len(items), cfg.shifts_per_batch)
    ]

    for batch in batches:
        start = time.perf_counter()
        sequence = _sequence(cfg, batch)
        postprocessors = [
            _postprocessor(cfg, sequence.t0, branch["shifts"])
            for _, branch in batch
        ]

        results = infer_local(client, sequence, postprocessors)
        for (branch_id, _), result, postproc in zip(
            batch, results, postprocessors, strict=True
        ):
            _write_outputs(
                outdir=scratch / branch_id,
                branch_id=branch_id,
                results=result,
                seq=sequence,
                postproc=postproc,
                cfg=cfg,
            )

        duration = sequence.stream_layout.size / cfg.sample_rate
        logging.info(
            f"Finished branches {[branch_id for branch_id, _ in batch]} "
            f"({duration:.0f} s of strain each) in "
            f"{time.perf_counter() - start:.1f} s"
        )


def _resolve_branches(cfg):
    """The group's branches, by ID, with the files they read.

    The `--branches` file, written by infer_group_branches in infer.smk,
    gives each branch's ID, shifts, and indices into `--background` and
    `--waveforms` (null for none).
    """
    with open(cfg.branches) as f:
        branch_list = json.load(f)
    branches = {}
    for branch in branch_list:
        index = branch["waveforms"]
        branches[str(branch["id"])] = {
            "fname": cfg.background[branch["background"]],
            "waveforms": None if index is None else cfg.waveforms[index],
            "shifts": branch["shifts"],
        }
    return branches


def _merge_group(cfg, branches, scratch):
    """Merge the group's branch outputs into the group's declared outputs.

    With `zero_lag_out`, branches with all-zero shifts go there rather
    than to `background_out`. The row count of each event set goes to
    `metadata_out`, so that aggregate_infer can size its outputs without
    opening every group's files an extra time.
    """
    split_zero_lag = cfg.zero_lag_out is not None
    background, zero_lag = [], []
    for branch_id, branch in branches.items():
        fname = scratch / branch_id / "background.hdf5"
        shifts = branch["shifts"]
        if split_zero_lag and all(s == 0 for s in shifts):
            zero_lag.append(fname)
        else:
            background.append(fname)
    # R&P foregrounds are plain EventSets, not RecoveredInjectionSets.
    foreground_cls = (
        EventSet if cfg.analysis_type == "rnp" else RecoveredInjectionSet
    )

    event_sets = {
        "background": (EventSet, background, cfg.background_out),
        "foreground": (
            foreground_cls,
            [scratch / i / "foreground.hdf5" for i in branches],
            cfg.foreground_out,
        ),
    }
    if split_zero_lag:
        event_sets["zero_lag"] = (EventSet, zero_lag, cfg.zero_lag_out)

    lengths = {}
    for name, (cls, files, fname) in event_sets.items():
        cls.aggregate(files, fname, clean=False)
        with h5py.File(fname, "r") as f:
            lengths[name] = int(f.attrs["length"])
    with open(cfg.metadata_out, "w") as f:
        json.dump(lengths, f)

    if cfg.timeseries_out is not None:
        merge_timeseries(
            [scratch / i / "timeseries.hdf5" for i in branches],
            cfg.timeseries_out,
        )


def _run_group(client, cfg, run_branches):
    """Run a group's branches with `run_branches`, then merge their outputs.

    `run_branches(client, cfg, branches, scratch)` writes each branch's
    outputs to `scratch / branch_id`.
    """
    branches = _resolve_branches(cfg)

    outdir = Path(cfg.background_out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=outdir) as scratch:
        scratch = Path(scratch)
        run_branches(client, cfg, branches, scratch)
        _merge_group(cfg, branches, scratch)
        logging.info(f"Wrote group outputs to {outdir}")


def _shared_args(p):
    p.add_argument("--config", action=jsonargparse.ActionConfigFile)
    p.add_argument("--verbose", type=bool, default=False)
    p.add_argument("--logfile", type=str, default=None)
    p.add_argument("--background", type=str, nargs="+")
    p.add_argument("--waveforms", type=str, nargs="*", default=[])
    p.add_argument("--branches", type=str)
    p.add_argument("--analysis_type", type=str, default="hdf5")
    p.add_argument("--background_out", type=str)
    p.add_argument("--foreground_out", type=str)
    p.add_argument("--metadata_out", type=str)
    p.add_argument("--zero_lag_out", type=str | None, default=None)
    p.add_argument("--timeseries_out", type=str | None, default=None)
    p.add_argument("--ifos", type=list[str])
    p.add_argument("--inference_sampling_rate", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--psd_length", type=float)
    p.add_argument("--fduration", type=float)
    p.add_argument("--integration_window_length", type=float)
    p.add_argument("--cluster_window_length", type=float)
    p.add_argument("--sample_rate", type=float)
    p.add_argument("--channel", type=str, default=None)  # R&P frames


def main_triton(args=None):
    p = jsonargparse.ArgumentParser()
    _shared_args(p)
    p.add_argument("--address", type=str)
    p.add_argument("--model_name", type=str)
    p.add_argument("--model_version", type=int, default=-1)
    p.add_argument("--rate", type=float | None, default=None)
    cfg = p.parse_args(args)
    if cfg.logfile is not None:
        Path(cfg.logfile).parent.mkdir(parents=True, exist_ok=True)
    configure_logging(cfg.logfile, verbose=cfg.verbose)

    from hermes.aeriel.client import InferenceClient

    client = InferenceClient(
        address=cfg.address,
        model_name=cfg.model_name,
        model_version=cfg.model_version,
        batch_size=cfg.batch_size,
    )
    _run_group(client, cfg, _run_triton)


def main_local(args=None):
    p = jsonargparse.ArgumentParser()
    _shared_args(p)
    p.add_argument("--weights", type=str)
    p.add_argument("--backend", type=str, default="export")
    p.add_argument("--aoti_path", type=str | None, default=None)
    p.add_argument("--kernel_length", type=float)
    p.add_argument("--highpass", type=float)
    p.add_argument("--fftlength", type=float | None, default=None)
    p.add_argument("--shifts_per_batch", type=int, default=1)
    cfg = p.parse_args(args)
    if cfg.logfile is not None:
        Path(cfg.logfile).parent.mkdir(parents=True, exist_ok=True)
    configure_logging(cfg.logfile, verbose=cfg.verbose)

    from infer.local import LocalInferenceClient

    logging.info(f"Loading model (backend={cfg.backend})")
    client = LocalInferenceClient(
        weights=cfg.weights,
        backend=cfg.backend,
        aoti_path=cfg.aoti_path,
        psd_length=cfg.psd_length,
        kernel_length=cfg.kernel_length,
        fduration=cfg.fduration,
        sample_rate=cfg.sample_rate,
        inference_sampling_rate=cfg.inference_sampling_rate,
        batch_size=cfg.batch_size,
        highpass=cfg.highpass,
        fftlength=cfg.fftlength,
    )
    _run_group(client, cfg, _run_local)
