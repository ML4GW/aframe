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
from pathlib import Path

import h5py
import jsonargparse
import numpy as np
from utils.logging import configure_logging

from infer.data import Hdf5Sequence, RnPSequence
from infer.main import infer
from infer.postprocess import Postprocessor


def _write_outputs(outdir, results, seq, postproc, return_timeseries):
    background, foreground, background_ts, foreground_ts = results
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    background.write(outdir / "background.hdf5")
    foreground.write(outdir / "foreground.hdf5")
    with (outdir / "metadata.json").open("w") as f:
        json.dump(
            {
                "background_length": len(background),
                "foreground_length": len(foreground),
            },
            f,
        )
    if return_timeseries:
        with h5py.File(outdir / "timeseries.hdf5", "w") as f:
            f.attrs["t0"] = seq.t0
            f.attrs["sample_t0"] = postproc.t0
            f.attrs["inference_sampling_rate"] = (
                postproc.inference_sampling_rate
            )
            f.attrs["shifts"] = postproc.shifts
            f.create_dataset("background", data=background_ts)
            f.create_dataset(
                "foreground",
                data=foreground_ts
                if foreground_ts is not None
                else np.zeros(0),
            )


def _run_branch(client, cfg, branch, outdir, rate=None):
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
            cfg.waveforms,
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
    _write_outputs(outdir, results, seq, postproc, cfg.return_timeseries)


def _run_group(client, cfg, rate=None, reset=None):
    with open(cfg.branch_map) as f:
        branch_map = json.load(f)
    ids = list(branch_map.keys())
    start = cfg.group_id * cfg.branches_per_job
    group = ids[start : start + cfg.branches_per_job]
    with client:
        for branch_id in group:
            if reset:
                reset()
            _run_branch(
                client,
                cfg,
                branch=branch_map[branch_id],
                outdir=Path(cfg.outdir_root) / branch_id,
                rate=rate,
            )


def _shared_args(p):
    p.add_argument("--config", action=jsonargparse.ActionConfigFile)
    p.add_argument("--verbose", type=bool, default=False)
    p.add_argument("--logfile", type=str, default=None)
    p.add_argument("--branch_map", type=str)
    p.add_argument("--group_id", type=int)
    p.add_argument("--branches_per_job", type=int)
    p.add_argument("--analysis_type", type=str, default="hdf5")
    p.add_argument("--waveforms", type=str)
    p.add_argument("--outdir_root", type=str)
    p.add_argument("--return_timeseries", type=bool, default=False)
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
