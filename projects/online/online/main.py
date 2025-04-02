import logging
from pathlib import Path
from queue import Empty
from typing import Iterable, List, Optional, Tuple

import torch
from amplfi.train.architectures.flows import FlowArchitecture
from amplfi.train.data.utils.utils import ParameterSampler
from architectures import Architecture
from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten
from torch.multiprocessing import Array, Process, Queue

from ledger.events import EventSet
from online.utils.buffer import InputBuffer, OutputBuffer
from online.utils.dataloading import data_iterator
from online.utils.gdb import GdbServer
from online.utils.ngdd import data_iterator as ngdd_data_iterator
from online.utils.pe import run_amplfi
from online.utils.searcher import Searcher
from online.utils.snapshotter import OnlineSnapshotter
from utils.preprocessing import BatchWhitener
from online.subprocesses import (
    amplfi_subprocess,
    pastro_subprocess,
    event_creation_subprocess,
    authenticate_subprocess,
)


SECONDS_PER_DAY = 86400


def load_model(model: Architecture, weights: Path):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
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
    aframe_right_pad: float,
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

    if aframe_right_pad > 0:
        time_offset -= kernel_length - aframe_right_pad
    elif aframe_right_pad < 0:
        # Trigger distance parameter accounts for fduration already
        time_offset -= abs(aframe_right_pad) - fduration / 2

    return time_offset


@torch.no_grad()
def search(
    whitener: BatchWhitener,
    snapshotter: OnlineSnapshotter,
    searcher: Searcher,
    event_queue: Queue,
    amplfi_queue: Queue,
    error_queue: Queue,
    input_buffer: InputBuffer,
    output_buffer: OutputBuffer,
    aframe: Architecture,
    amplfi_hl: FlowArchitecture,
    scaler_hl: ChannelWiseScaler,
    amplfi_hlv: FlowArchitecture,
    scaler_hlv: ChannelWiseScaler,
    spectral_density: SpectralDensity,
    amplfi_whitener: Whiten,
    samples_per_event: int,
    shared_samples: Array,
    data_it: Iterable[Tuple[torch.Tensor, float, bool]],
    update_size: float,
    time_offset: float,
    device: str,
):
    integrated = None

    # flag that declares if the most previous frame
    # was analysis ready or not
    in_spec = False

    virgo_ready = [False] * (input_buffer.buffer_length // update_size)

    state = snapshotter.initial_state
    for X, t0, ready in data_it:
        # TODO:
        # here we can handle any subprocess
        # errors - I think at the least we
        # need to send an email that
        # something failed and the pipeline
        # needs to be restarted
        try:
            name, error, tb = error_queue.get_nowait()
        except Empty:
            pass
        else:
            logging.error(f"Error in subprocess {name}: {error}")
            logging.error(tb)

        # if this frame was not analysis ready, assuming HLV ordering
        # on ready array
        hl_ready = ready[0] and ready[1]
        virgo_ready.append(ready[-1])
        virgo_ready.pop(0)
        if not hl_ready:
            if searcher.detecting:
                # if we were in the middle of a detection,
                # we won't get to see the peak of the event
                # so build the event with what we have
                event = searcher.build_event(
                    integrated[-1], t0 - update_size, len(integrated) - 1
                )
                if event is not None:
                    # maybe process event found in the previous frame
                    logging.info("Putting event in event queue")
                    event_queue.put(event)
                    logging.info("Running AMPLFI")
                    if all(virgo_ready) and len(ready) == 3:
                        logging.info("Using HLV AMPLFI model")
                        amplfi = amplfi_hlv
                        scaler = scaler_hlv
                    else:
                        logging.info("Using HL AMPLFI model")
                        amplfi = amplfi_hl
                        scaler = scaler_hl
                    descaled_samples = run_amplfi(
                        event_time=event.gpstime,
                        input_buffer=input_buffer,
                        samples_per_event=samples_per_event,
                        spectral_density=spectral_density,
                        amplfi_whitener=amplfi_whitener,
                        amplfi=amplfi,
                        std_scaler=scaler,
                        device=device,
                    )
                    for i, sample in enumerate(descaled_samples.flatten()):
                        shared_samples[i] = sample
                    amplfi_queue.put(event.gpstime)
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

        # update our input buffer with latest strain data
        input_buffer.update(X, t0)

        # we have a frame that is analysis ready,
        # so lets analyze it:
        X = X[:2].to(device)

        # update the snapshotter state and return
        # unfolded batch of overlapping windows
        batch, state = snapshotter(X[None], state)

        # whiten the batch, and analyze with aframe
        whitened = whitener(batch)
        y = aframe(whitened)[:, 0]

        # update our output buffer with the latest aframe output,
        # which will also automatically integrate the output
        integrated = output_buffer.update(y.cpu(), t0)

        # if this frame was analysis ready,
        # and we had enough previous to build whitening filter
        # search for events in the integrated output
        event = None
        if snapshotter.full_psd_present and hl_ready:
            event = searcher.search(integrated, t0 + time_offset)

        # if we found an event, process it!
        if event is not None:
            logging.info("Putting event in event queue")
            event_queue.put(event)
            logging.info("Running AMPLFI")
            if all(virgo_ready) and len(ready) == 3:
                logging.info("Using HLV AMPLFI model")
                amplfi = amplfi_hlv
                scaler = scaler_hlv
            else:
                logging.info("Using HL AMPLFI model")
                amplfi = amplfi_hl
                scaler = scaler_hl
            descaled_samples = run_amplfi(
                event_time=event.gpstime,
                input_buffer=input_buffer,
                samples_per_event=samples_per_event,
                spectral_density=spectral_density,
                amplfi_whitener=amplfi_whitener,
                amplfi=amplfi,
                std_scaler=scaler,
                device=device,
            )
            for i, sample in enumerate(descaled_samples.flatten()):
                shared_samples[i] = sample
            amplfi_queue.put(event.gpstime)
            searcher.detecting = False
        # TODO write buffers to disk:


def main(
    aframe_weights: Path,
    amplfi_hl_architecture: FlowArchitecture,
    amplfi_hl_weights: Path,
    amplfi_hlv_architecture: FlowArchitecture,
    amplfi_hlv_weights: Path,
    amplfi_parameter_sampler: ParameterSampler,
    background_path: Path,
    foreground_path: Path,
    rejected_path: Path,
    outdir: Path,
    datadir: Path,
    ifos: List[str],
    inference_params: List[str],
    channels: List[str],
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    psd_length: float,
    aframe_right_pad: float,
    amplfi_kernel_length: float,
    event_position: float,
    fduration: float,
    integration_window_length: float,
    astro_event_rate: float,
    data_source: str = "frames",
    state_channels: Optional[dict[str, str]] = None,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    lowpass: Optional[float] = None,
    refractory_period: float = 8,
    far_threshold: float = 1,
    server: GdbServer = "test",
    ifo_suffix: str = None,
    input_buffer_length: int = 75,
    output_buffer_length: int = 8,
    samples_per_event: int = 20000,
    nside: int = 32,
    device: str = "cpu",
):
    """
    Main function for launching real-time Aframe and AMPLFI pipeline.

    Args:
        aframe_weights:
            Path to trained Aframe model weights
        amplfi_architecture:
            AMPLFI model architecture for parameter estimation
        amplfi_weights:
            Path to trained AMPLFI model weights
        background_path:
            Path to background noise events dataset
            used for Aframe FAR calculation
        foreground_path:
            Path to recovered injection events dataset
        rejected_path:
            Path to rejected injection parameters dataset
        outdir:
            Directory to save output files
        datadir:
            Directory containing input strain data
        ifos:
            List of interferometer names
        inference_params:
            List of parameters on which the AMPLFI
            model was trained to perform inference
        channels:
            List of the channel names to analyze
        sample_rate:
            Input data sample rate in Hz
        kernel_length:
            Length of Aframe analysis kernel in seconds
        inference_sampling_rate:
            Rate at which to sample the output of the Aframe model
        psd_length:
            Length of PSD estimation window in seconds
        aframe_right_pad:
            Time offset for trigger positioning in seconds
        amplfi_kernel_length:
            Length of AMPLFI analysis window in seconds
        event_position:
            Event position (in seconds) from the left edge
            of the analysis window used for parameter estimation
        fduration:
            Length of whitening filter in seconds
        integration_window_length:
            Length of output integration window in seconds
        astro_event_rate:
            Prior on rate of astrophysical events in units Gpc^-3 yr^-1
        fftlength:
            FFT length in seconds (defaults to kernel_length + fduration)
        highpass:
            High-pass filter frequency in Hz
        lowpass:
            Low-pass filter frequency in Hz
        refractory_period:
            Minimum time between events in seconds
        far_threshold:
            False alarm rate threshold in events/day
        server:
            GraceDB server to use:
            "local", "playground", "test" or "production"
        ifo_suffix:
            Optional suffix for accessing data from /dev/shm.
            Useful when analyzing alternative streams like
            MDC replays that have directory structures in the format
            `{ifo}_{ifo_suffix}`
        input_buffer_length:
            Length of strain data buffer in seconds
        output_buffer_length:
            Length of inference output buffer in seconds
        samples_per_event:
            Number of posterior samples to generate per event
            for creating skymaps and other parameter estimation
            data products
        nside:
            Healpix resolution for low-latency skymaps
        device:
            Device to run inference on ("cpu" or "cuda")
    """

    if ifos not in [["H1", "L1"], ["H1", "L1", "V1"]]:
        raise ValueError(
            f"Invalid interferometer configuration {ifos}. "
            "Must be ['H1', 'L1'] or ['H1', 'L1', 'V1']"
        )

    fftlength = fftlength or kernel_length + fduration

    # initialize multiprocessing Array that will be used
    # to pass amplfi samples between different subprocesses
    data = torch.randn(samples_per_event * len(inference_params))
    shared_samples = Array("d", data)

    # create various queues for message
    # passing between subprocesses
    error_queue = Queue()
    pastro_queue = Queue()
    event_queue = Queue()
    amplfi_queue = Queue()

    # subprocess for re-authenticating
    args = (error_queue, "authenticate")
    auth_process = Process(target=authenticate_subprocess, args=args)
    auth_process.start()

    # create subprocess for uploading initial
    # detection information like FAR to gdb
    args = (
        error_queue,
        "event creator",
        event_queue,
        server,
        outdir / "events",
        amplfi_queue,
        pastro_queue,
    )
    event_process = Process(target=event_creation_subprocess, args=args)
    event_process.start()

    # initialize amplfi subprocess which
    # will recieve events via a queue
    # and process posterior samples into
    # various data products and upload them
    # to gdb

    args = (
        error_queue,
        "amplfi",
        amplfi_queue,
        server,
        outdir / "events",
        inference_params,
        amplfi_parameter_sampler,
        shared_samples,
        nside,
    )

    amplfi_process = Process(target=amplfi_subprocess, args=args)
    amplfi_process.start()

    # create a subprocess for calculating
    # and uploading pastro to gdb
    # once events are detected

    args = (
        error_queue,
        "p_astro",
        pastro_queue,
        background_path,
        foreground_path,
        rejected_path,
        astro_event_rate,
        server,
        outdir,
    )
    pastro_process = Process(target=pastro_subprocess, args=args)
    pastro_process.start()

    if state_channels is None:
        logging.info(
            "No state channels specified: not checking for data quality"
        )

    if data_source == "ngdd":
        update_size = 1 / 16
        data_it = ngdd_data_iterator(
            strain_channels=channels,
            ifos=ifos,
            sample_rate=sample_rate,
            state_channels=state_channels,
        )
    elif data_source == "frames":
        update_size = 1
        data_it = data_iterator(
            datadir=datadir,
            channels=channels,
            ifos=ifos,
            sample_rate=sample_rate,
            ifo_suffix=ifo_suffix,
            state_channels=state_channels,
            timeout=10,
        )
    else:
        raise ValueError(
            f"Invalid data source {data_source}. Must be 'ngdd' or 'frames'"
        )

    # initialize a buffer for storing recent strain data,
    # and for storing integrated aframe outputs
    input_buffer = InputBuffer(
        ifos=ifos,
        sample_rate=sample_rate,
        buffer_length=input_buffer_length,
        fduration=fduration,
        amplfi_kernel_length=amplfi_kernel_length,
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
    aframe = torch.jit.load(aframe_weights)
    aframe = aframe.to(device)

    logging.info(f"Loading HL AMPLFI from weights at path {amplfi_hl_weights}")
    amplfi_hl, scaler_hl = load_amplfi(
        amplfi_hl_architecture, amplfi_hl_weights, len(inference_params)
    )
    amplfi_hl = amplfi_hl.to(device)
    scaler_hl = scaler_hl.to(device)

    logging.info(
        f"Loading HLV AMPLFI from weights at path {amplfi_hlv_weights}"
    )
    amplfi_hlv, scaler_hlv = load_amplfi(
        amplfi_hlv_architecture, amplfi_hlv_weights, len(inference_params)
    )
    amplfi_hlv = amplfi_hlv.to(device)
    scaler_hlv = scaler_hlv.to(device)

    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        average="median",
    ).to(device)
    amplfi_whitener = Whiten(
        fduration=fduration,
        sample_rate=sample_rate,
        highpass=highpass,
        lowpass=lowpass,
    ).to(device)

    whitener = BatchWhitener(
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
        batch_size=update_size * inference_sampling_rate,
        fduration=fduration,
        fftlength=fftlength,
        highpass=highpass,
        lowpass=lowpass,
    ).to(device)

    # Hard-coding number of channels until Aframe is generalized
    snapshotter = OnlineSnapshotter(
        update_size=update_size,
        num_channels=2,
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
    ).to(device)

    # load in background, foreground, and rejected injections;
    # use them to fit (or load in a cached) a pastro model
    logging.info("Loading background distribution")
    background = EventSet.read(background_path)

    # if event set is not sorted by detection statistic, sort it
    # which will significantly speed up the far calculation
    if not background.is_sorted_by("detection_statistic"):
        logging.info(
            "Sorting background by detection statistic and writing to disk"
        )
        background = background.sort_by("detection_statistic")
        background.write(background_path)

    # Convert FAR to Hz from 1/days
    far_threshold /= SECONDS_PER_DAY
    searcher = Searcher(
        background=background,
        far_threshold=far_threshold,
        inference_sampling_rate=inference_sampling_rate,
        refractory_period=refractory_period,
        ifos=ifos,
        channels=channels,
        datadir=datadir,
        ifo_suffix=ifo_suffix,
    )

    time_offset = get_time_offset(
        inference_sampling_rate,
        fduration,
        integration_window_length,
        kernel_length,
        aframe_right_pad,
    )

    logging.info("Beginning search...")
    search(
        whitener=whitener,
        snapshotter=snapshotter,
        searcher=searcher,
        error_queue=error_queue,
        event_queue=event_queue,
        amplfi_queue=amplfi_queue,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        aframe=aframe,
        amplfi_hl=amplfi_hl,
        scaler_hl=scaler_hl,
        amplfi_hlv=amplfi_hlv,
        scaler_hlv=scaler_hlv,
        spectral_density=spectral_density,
        amplfi_whitener=amplfi_whitener,
        samples_per_event=samples_per_event,
        shared_samples=shared_samples,
        data_it=data_it,
        update_size=update_size,
        time_offset=time_offset,
        device=device,
    )


if __name__ == "__main__":
    main()
