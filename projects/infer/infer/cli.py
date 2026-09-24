"""Inference CLI for groups of branches.

Two entry points:
    infer-triton: stream branches to a running Triton server.
    infer-local: run branches in-process on a GPU.

Both use the same main.infer() loop and Postprocessor, so outputs are
identical and aggregate_infer is client agnostic.

--analysis_type selects the Sequence: hdf5 (timeslide background + injections)
or rnp (Rates and Populations frames).
"""

import json
import tempfile
from pathlib import Path

import h5py
import jsonargparse
from ledger.events import EventSet, RecoveredInjectionSet
from utils.logging import configure_logging

from infer.aggregate import merge_timeseries, write_timeseries
from infer.data import Hdf5Sequence, RnPSequence
from infer.main import infer
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
            sample_t0=seq.t0 - cfg.fduration / 2,
            inference_sampling_rate=postproc.inference_sampling_rate,
            shifts=postproc.shifts,
        )


def _run_branch(client, cfg, branch_id, branch, outdir, rate=None):
    shifts = branch["shifts"]
    if cfg.analysis_type == "rnp":
        seq = RnPSequence(
            injection_file=Path(branch["fname"]),
            channel=cfg.channel,
            ifos=cfg.ifos,
            sample_rate=cfg.sample_rate,
            inference_sampling_rate=cfg.inference_sampling_rate,
            batch_size=cfg.batch_size,
        )
    else:
        seq = Hdf5Sequence(
            branch["fname"],
            branch["waveforms"],
            cfg.ifos,
            shifts,
            cfg.inference_sampling_rate,
            cfg.batch_size,
            rate=rate,
        )
    client.callback = seq
    postproc = Postprocessor(
        t0=seq.t0,
        shifts=shifts,
        psd_length=cfg.psd_length,
        fduration=cfg.fduration,
        inference_sampling_rate=cfg.inference_sampling_rate,
        integration_window_length=cfg.integration_window_length,
        cluster_window_length=cfg.cluster_window_length,
    )
    results = infer(client, seq, postproc)
    _write_outputs(outdir, branch_id, results, seq, postproc, cfg)


def _merge_group(cfg, branch_map, group, scratch):
    """Merge the group's branch outputs into the group's declared outputs.

    With `zero_lag_out`, branches with all-zero shifts go there rather
    than to `background_out`. The row count of each event set goes to
    `metadata_out`, so that aggregate_infer can size its outputs without
    opening every group's files an extra time.
    """
    split_zero_lag = cfg.zero_lag_out is not None
    background, zero_lag = [], []
    for branch_id in group:
        fname = scratch / branch_id / "background.hdf5"
        shifts = branch_map[branch_id]["shifts"]
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
            [scratch / i / "foreground.hdf5" for i in group],
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
            [scratch / i / "timeseries.hdf5" for i in group],
            cfg.timeseries_out,
        )


def _run_group(client, cfg, rate=None, reset=None):
    with open(cfg.branch_map) as f:
        branch_map = json.load(f)
    # groups are assigned by compute_branch_map in infer.smk
    group = [i for i, b in branch_map.items() if b["group"] == cfg.group_id]

    # Per-branch outputs only exist until they're merged. Keep them next
    # to the group's outputs rather than in /tmp, which may be small.
    outdir = Path(cfg.background_out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=outdir) as scratch:
        scratch = Path(scratch)
        with client:
            for branch_id in group:
                if reset:
                    reset()
                _run_branch(
                    client,
                    cfg,
                    branch_id=branch_id,
                    branch=branch_map[branch_id],
                    outdir=scratch / branch_id,
                    rate=rate,
                )
        _merge_group(cfg, branch_map, group, scratch)


def _shared_args(p):
    p.add_argument("--config", action=jsonargparse.ActionConfigFile)
    p.add_argument("--verbose", type=bool, default=False)
    p.add_argument("--logfile", type=str, default=None)
    p.add_argument("--branch_map", type=str)
    p.add_argument("--group_id", type=int)
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
    _run_group(client, cfg, rate=cfg.rate)


def main_local(args=None):
    p = jsonargparse.ArgumentParser()
    _shared_args(p)
    p.add_argument("--weights", type=str)
    p.add_argument("--backend", type=str, default="export")
    p.add_argument("--aoti_path", type=str | None, default=None)
    p.add_argument("--sample_rate", type=float)
    p.add_argument("--kernel_length", type=float)
    p.add_argument("--highpass", type=float)
    p.add_argument("--fftlength", type=float | None, default=None)
    p.add_argument("--channel", type=str, default=None)  # R&P frames
    cfg = p.parse_args(args)
    if cfg.logfile is not None:
        Path(cfg.logfile).parent.mkdir(parents=True, exist_ok=True)
    configure_logging(cfg.logfile, verbose=cfg.verbose)

    from infer.local import LocalInferenceClient

    client = LocalInferenceClient(
        weights=cfg.weights,
        backend=cfg.backend,
        aoti_path=cfg.aoti_path,
        num_ifos=len(cfg.ifos),
        psd_length=cfg.psd_length,
        kernel_length=cfg.kernel_length,
        fduration=cfg.fduration,
        sample_rate=cfg.sample_rate,
        inference_sampling_rate=cfg.inference_sampling_rate,
        batch_size=cfg.batch_size,
        highpass=cfg.highpass,
        fftlength=cfg.fftlength,
    )
    _run_group(client, cfg, reset=client.reset)
