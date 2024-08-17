import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from architectures import Architecture
from ledger.events import EventSet
from ligo.gracedb.rest import GraceDb
from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten
from online.utils.buffer import InputBuffer, OutputBuffer
from online.utils.dataloading import data_iterator
from online.utils.gdb import gracedb_factory
from online.utils.pe import run_amplfi
from online.utils.searcher import Event, Searcher
from online.utils.snapshotter import OnlineSnapshotter

from amplfi.architectures.flows.base import FlowArchitecture
from utils.preprocessing import BatchWhitener

# seconds of data per update
UPDATE_SIZE = 1


def load_model(model: Architecture, weights: Path):
    checkpoint = torch.load(weights, map_location="cpu")
    arch_weights = {
        k[6:]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(arch_weights)
    model.eval()
    return model, checkpoint


def load_amplfi(model: FlowArchitecture, weights: Path, num_params: int):
    model, checkpoint = load_model(model, weights)
    scaler_weights = {
        k[len("scaler.") :]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("scaler.")
    }
    scaler = ChannelWiseScaler(num_params)
    scaler.load_state_dict(scaler_weights)
    return model, scaler


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
    amplfi: FlowArchitecture,
    scaler: ChannelWiseScaler,
    outdir: Path,
    device: str,
):
    event.write(outdir)
    response = gdb.submit(event)
    last_event_time = event.gpstime
    posterior, skymap = run_amplfi(
        last_event_time,
        buffer,
        spectral_density,
        pe_whitener,
        amplfi,
        scaler,
        outdir / "whitened_data_plots",
        device,
    )
    graceid = response.json()["graceid"]
    gdb.submit_pe(posterior, skymap, graceid)
    pass


@torch.no_grad()
def search(
    gdb: GraceDb,
    pe_whitener: Whiten,
    scaler: torch.nn.Module,
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
    device: str,
):
    integrated = None

    # flat that declares if the most previous frame
    # was analysis ready or not
    in_spec = False

    #
    state = snapshotter.initial_state
    for X, t0, ready in data_it:
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
                        scaler,
                        outdir,
                        device,
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

                input_buffer.reset()
                output_buffer.reset()

                # nothing left to do, so move on to next frame
                continue

        elif not in_spec:
            # the frame is analysis ready, but previous frames
            # weren't, so reset our running states
            logging.info(f"Frame {t0} is ready again, resetting states")
            state = snapshotter.reset()
            input_buffer.reset()
            output_buffer.reset()
            in_spec = True

        # we have a frame that is analysis ready,
        # so lets analyze it:
        X = X.to(device)

        # update the snapshotter state and return
        # unfolded batch of overlapping windows
        batch, state = snapshotter(X[None], state)

        # whiten the batch, and analyze with aframe
        whitened = whitener(batch)
        y = aframe(whitened)[:, 0]

        # update our input buffer with latest strain data,
        input_buffer.update(X.cpu(), t0)
        # update our output buffer with the latest aframe output,
        # which will also automatically integrate the output
        integrated = output_buffer.update(y.cpu(), t0)

        # if this frame was analysis ready,
        # and we had enough previous to build whitening filter
        # search for events in the integrated output
        event = None
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
                scaler,
                outdir,
                device,
            )
            searcher.detecting = False

        # TODO write buffers to disk:


def main(
    aframe_weights: Path,
    amplfi_architecture: FlowArchitecture,
    amplfi_weights: Path,
    background_file: Path,
    outdir: Path,
    datadir: Path,
    ifos: List[str],
    inference_params: List[str],
    channel: str,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    psd_length: float,
    trigger_distance: float,
    pe_window: float,
    event_position: float,
    fduration: float,
    integration_window_length: float,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    refractory_period: float = 8,
    far_threshold: float = 1,
    server: str = "test",
    ifo_suffix: str = None,
    input_buffer_length: int = 75,
    output_buffer_length: int = 8,
    device: str = "cpu",
):
    gdb = gracedb_factory(server, outdir)

    # initialize a buffer for storing recent strain data,
    # and for storing integrated aframe outputs
    input_buffer = InputBuffer(
        ifos=ifos,
        sample_rate=sample_rate,
        buffer_length=input_buffer_length,
        fduration=fduration,
        pe_window=pe_window,
        event_position=event_position,
        device="cpu",
    )

    output_buffer = OutputBuffer(
        inference_sampling_rate=inference_sampling_rate,
        integration_window_length=integration_window_length,
        buffer_length=output_buffer_length,
        device="cpu",
    )

    # Load in Aframe and amplfi models
    logging.info(f"Loading Aframe from weights at path {aframe_weights}")
    logging.info(f"Loading AMPLFI from weights at path {amplfi_weights}")
    aframe = torch.jit.load(aframe_weights)
    aframe = aframe.to(device)
    amplfi, scaler = load_amplfi(
        amplfi_architecture, amplfi_weights, len(inference_params)
    )
    amplfi = amplfi.to(device)
    scaler = scaler.to(device)

    fftlength = fftlength or kernel_length + fduration

    whitener = BatchWhitener(
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
        batch_size=UPDATE_SIZE * inference_sampling_rate,
        fduration=fduration,
        fftlength=fftlength,
        highpass=highpass,
    ).to(device)

    snapshotter = OnlineSnapshotter(
        update_size=UPDATE_SIZE,
        num_channels=len(ifos),
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
    ).to(device)

    # Amplfi setup. Hard code most of it for now
    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        average="median",
    ).to(device)
    pe_whitener = Whiten(
        fduration=fduration, sample_rate=sample_rate, highpass=highpass
    ).to(device)

    background = EventSet.read(background_file)

    # Convert FAR to Hz from 1/days
    far_threshold /= 86400
    searcher = Searcher(
        background,
        far_threshold,
        inference_sampling_rate,
        refractory_period,
        ifos,
        datadir,
        ifo_suffix,
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
        gdb=gdb,
        pe_whitener=pe_whitener,
        scaler=scaler,
        spectral_density=spectral_density,
        whitener=whitener,
        snapshotter=snapshotter,
        searcher=searcher,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        aframe=aframe,
        amplfi=amplfi,
        data_it=data_it,
        time_offset=time_offset,
        outdir=outdir,
        device=device,
    )


if __name__ == "__main__":
    main()
