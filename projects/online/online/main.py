import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from architectures import Architecture
from ledger.events import EventSet
from ligo.gracedb.rest import GraceDb
from online.buffer import InputBuffer, OutputBuffer
from online.dataloading import data_iterator
from online.gdb import gracedb_factory
from online.pe import run_amplfi
from online.searcher import Event, Searcher
from online.snapshotter import OnlineSnapshotter

from ml4gw.transforms import SpectralDensity, Whiten
from utils.preprocessing import BatchWhitener

# seconds of data per update
UPDATE_SIZE = 1


def load_model(model: Architecture, weights: Path):
    checkpoint = torch.load(weights)
    arch_weights = {
        k: v for k, v in checkpoint.items() if k.startswith("model.")
    }
    model.load_state_dict(arch_weights)
    model.to("cuda")
    model.eval()
    return model


def get_time_offset(
    inference_sampling_rate: float,
    fduration: float,
    integration_window_length: float,
    kernel_length: float,
    trigger_distance: float,
):
    # offset the initial timestamp of our
    # integrated outputs relative to the
    # initial timestamp of the most recently
    # loaded frames
    time_offset = (
        1 / inference_sampling_rate  # end of the first kernel in batch
        - fduration / 2  # account for whitening padding
        - integration_window_length  # account for time to build peak
    )

    if trigger_distance is not None:
        if trigger_distance > 0:
            time_offset -= kernel_length - trigger_distance
        if trigger_distance < 0:
            # Trigger distance parameter accounts for fduration already
            time_offset -= np.abs(trigger_distance) - fduration / 2

    return time_offset


def process_event(
    event: Event,
    gdb: GraceDb,
    buffer: InputBuffer,
    spectral_density: SpectralDensity,
    pe_whitener: Whiten,
    amplfi: torch.nn.Module,
    std_scaler: torch.nn.Module,
    outdir: Path,
):
    response = gdb.submit(event)
    last_event_time = event.gpstime
    posterior, skymap = run_amplfi(
        last_event_time,
        buffer,
        spectral_density,
        pe_whitener,
        amplfi,
        std_scaler,
        outdir / "whitened_data_plots",
    )
    graceid = response.json()["graceid"]
    gdb.submit_pe(posterior, skymap, graceid)
    pass


def search(
    gdb: GraceDb,
    pe_whitener: Whiten,
    std_scaler: torch.nn.Module,
    spectral_density: SpectralDensity,
    whitener: BatchWhitener,
    snapshotter: OnlineSnapshotter,
    searcher: Searcher,
    input_buffer: InputBuffer,
    output_buffer: OutputBuffer,
    aframe: Architecture,
    amplfi: Architecture,
    data_it: Iterable[Tuple[torch.Tensor, float, bool]],
    time_offset: float,
    outdir: Path,
):
    integrated = None
    in_spec = False
    state = snapshotter.initial_state
    for X, t0, ready in data_it:
        X = X.to("cuda")

        # if this frame was not analysis ready
        if not ready:
            if searcher.detecting:
                # if we were in the middle of a detection,
                # we won't get to see the peak of the event
                # so build the event with what we have
                event = searcher.build_event(
                    integrated, t0 - 1, len(integrated) - 1
                )
                if event is not None:
                    # maybe process event found in the previous frame
                    process_event(
                        event,
                        gdb,
                        input_buffer,
                        spectral_density,
                        pe_whitener,
                        amplfi,
                        std_scaler,
                        outdir,
                    )
                    searcher.detecting = False

            # check if this is because the frame stream stopped
            # being analysis ready, in which case perform updates
            # but don't search for events
            if X is not None:
                logging.warning(
                    "Frame {} is not analysis ready. Performing "
                    "inference but ignoring any triggers".format(t0)
                )
            # or if it's because frames were dropped within the stream
            # in which case we should reset our states
            else:
                logging.warning(
                    "Missing frame files after timestep {}, "
                    "resetting states".format(t0)
                )

                input_buffer.reset_state()
                output_buffer.reset_state()

                # nothing left to do, so move on to next frame
                continue

        elif not in_spec:
            # the frame is analysis ready, but previous frames
            # weren't, so reset our running states
            logging.info(f"Frame {t0} is ready again, resetting states")
            state = snapshotter.reset()
            input_buffer.reset_state()
            output_buffer.reset_state()
            in_spec = True

        # we have a frame that is analysis ready,
        # so lets analyze it:

        # update the snapshotter state and return
        # unfolded batch of overlapping windows
        batch, state = snapshotter(X, state)

        # whiten the batch, and analyze with aframe
        whitened = whitener(batch)
        y = aframe(whitened)[:, 0]

        # update our input buffer with latest strain data,
        input_buffer.update(X)
        # update our output buffer with the latest aframe output,
        # which will also automatically integrate the output
        integrated = output_buffer.update(y, t0)

        # if this frame was analysis ready,
        # and we had enough previous to build whitening filter
        # search for events in the integrated output
        if snapshotter.full_psd_present and ready:
            event = searcher.search(integrated, t0 + time_offset)

        # if we found an event, process it!
        if event is not None:
            process_event(
                event,
                gdb,
                input_buffer,
                spectral_density,
                pe_whitener,
                amplfi,
                std_scaler,
                outdir,
            )
            searcher.detecting = False

        # TODO write buffers to disk:


def main(
    aframe_architecture: Architecture,
    aframe_weights: Path,
    amplfi_architecture: Architecture,
    amplfi_weights: Path,
    background_file: Path,
    outdir: Path,
    datadir: Path,
    ifos: List[str],
    channel: str,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    psd_length: float,
    trigger_distance: float,
    fduration: float,
    integration_window_length: float,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    refractory_period: float = 8,
    far_threshold: float = 1,
    server: str = "test",
    ifo_suffix: str = None,
    input_buffer_length=75,
    output_buffer_length=8,
    verbose: bool = False,
):
    gdb = gracedb_factory(server)
    num_ifos = len(ifos)

    # initialize a buffer for storing recent strain data,
    # and for storing integrated aframe outputs
    input_buffer = InputBuffer(
        num_channels=num_ifos,
        sample_rate=sample_rate,
        buffer_length=input_buffer_length,
        fduration=fduration,
    )
    output_buffer = OutputBuffer(
        inference_sampling_rate=inference_sampling_rate,
        integration_window_length=integration_window_length,
        buffer_length=output_buffer_length,
    )

    # Load in Aframe and amplfi models
    logging.info(
        f"Loading Aframe from weights at path {aframe_weights}\n"
        f"Loading AMPLFI from weights at path {amplfi_weights}"
    )
    aframe = load_model(aframe_architecture, aframe_weights)
    amplfi = load_model(amplfi_architecture, amplfi_weights)

    fftlength = fftlength or kernel_length + fduration

    whitener = BatchWhitener(
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
        batch_size=UPDATE_SIZE * inference_sampling_rate,
        fduration=fduration,
        fftlength=fftlength,
        highpass=highpass,
    )

    snapshotter = OnlineSnapshotter(
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
    )

    # Amplfi setup. Hard code most of it for now
    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        average="median",
    ).to("cuda")
    pe_whitener = Whiten(
        fduration=fduration, sample_rate=sample_rate, highpass=highpass
    ).to("cuda")

    background = EventSet.read(background_file)

    searcher = Searcher(
        background, far_threshold, inference_sampling_rate, refractory_period
    )

    time_offset = get_time_offset(
        inference_sampling_rate,
        fduration,
        integration_window_length,
        kernel_length,
        trigger_distance,
    )

    data_it = data_iterator(
        datadir=datadir,
        channel=channel,
        ifos=ifos,
        sample_rate=sample_rate,
        ifo_suffix=ifo_suffix,
        timeout=10,
    )

    search(
        gdb,
        pe_whitener,
        None,
        spectral_density,
        whitener,
        snapshotter,
        searcher,
        input_buffer,
        output_buffer,
        aframe,
        amplfi,
        data_it,
        outdir,
        time_offset,
    )


if __name__ == "__main__":
    main()
