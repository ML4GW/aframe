import atexit
import numpy as np
import logging
import signal
from pathlib import Path
from queue import Empty
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING, Literal
from online.utils.email_alerts import send_error_email, send_init_email
import torch
from amplfi.train.architectures.flows import FlowArchitecture
from amplfi.train.data.utils.utils import ParameterSampler
from architectures import Architecture
from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten
from torch.multiprocessing import Array, Process, Queue

from ledger.events import EventSet
from online.utils.buffer import InputBuffer, OutputBuffer
from online.utils.dataloading import data_iterator
from online.utils.gdb import gracedb_factory
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
    cleanup_subprocesses,
    signal_handler,
)
from online.subprocesses.authenticate import authenticate

if TYPE_CHECKING:
    from online.utils.gdb import GdbServer

SECONDS_PER_DAY = 86400
# igwn_auth_utils finds tokens only if they have at least 10 minutes left
MIN_VALID_LIFETIME = 600
# 3 hours; not sure where this is documented
SCITOKEN_LIFETIME = 10800


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
    online_inference_rate: float,
    fduration: float,
    integration_window_length: float,
    kernel_length: float,
    aframe_right_pad: float,
):
    time_offset = (
        # end of the first kernel in batch
        1 / online_inference_rate
        # account for whitening padding
        - fduration / 2
        # distance coalescence time lies away from right edge
        - aframe_right_pad
        # account for time to build peak
        - integration_window_length
    )

    return time_offset


def process_event(
    event_queue: Queue,
    amplfi_queue: Queue,
    event: Queue,
    ifos_to_model: dict[
        tuple[str, ...], tuple[FlowArchitecture, ChannelWiseScaler]
    ],
    ifos: list[str],
    input_buffer: InputBuffer,
    output_buffer: OutputBuffer,
    samples_per_event: int,
    spectral_density: SpectralDensity,
    amplfi_whitener: Whiten,
    psd_length: float,
    shared_samples: Array,
    outdir: Path,
    device: str,
):
    logging.info("Putting event in event queue")
    event_queue.put(event)

    logging.info(f"Using {','.join(ifos)} AMPLFI model")
    amplfi, scaler = ifos_to_model[tuple(ifos)]

    amplfi_psd_strain, amplfi_strain = input_buffer.get_amplfi_data(
        event.gpstime, ifos, psd_length
    )
    descaled_samples, whitened, asds, freqs = run_amplfi(
        amplfi_strain,
        amplfi_psd_strain,
        samples_per_event=samples_per_event,
        spectral_density=spectral_density,
        amplfi_whitener=amplfi_whitener,
        amplfi=amplfi,
        std_scaler=scaler,
        device=device,
    )
    shared_samples[:] = descaled_samples.cpu().numpy().flatten()
    amplfi_queue.put((event, ifos))

    # save nn output, amplfi psds, and amplfi whitened strain
    buffer_outdir = outdir / "events" / event.event_dir
    buffer_outdir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Writing output buffer to {buffer_outdir}")
    output_buffer.write(buffer_outdir / "output.hdf5")

    asd = torch.cat((freqs[None][None].cpu(), asds.cpu()), dim=1).numpy()
    np.save(buffer_outdir / "asd.npy", asd)
    np.save(buffer_outdir / "amplfi_whitened.npy", whitened.cpu())


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
    ifos_to_model: dict[
        tuple[str, ...], tuple[FlowArchitecture, ChannelWiseScaler]
    ],
    spectral_density: SpectralDensity,
    amplfi_psd_length: float,
    amplfi_whitener: Whiten,
    samples_per_event: int,
    shared_samples: Array,
    data_it: Iterable[Tuple[torch.Tensor, float, bool]],
    update_size: float,
    time_offset: float,
    device: str,
    outdir: Path,
    emails: Optional[list[str]] = None,
):
    significance_outputs, timing_outputs = None, None

    # flag that declares if the most previous frame
    # was analysis ready or not
    in_spec = False

    virgo_ready = [False] * (input_buffer.buffer_length // update_size)

    state = snapshotter.initial_state
    for X, t0, ready in data_it:
        # handle any subprocess
        try:
            name, error, tb = error_queue.get_nowait()
        except Empty:
            pass
        else:
            logging.error(f"Error in subprocess {name}: {str(error)}")
            logging.error(tb)
            if emails is not None:
                send_error_email(name, error, tb, emails)
            raise error

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
                    significance_outputs[-1],
                    t0 - update_size,
                    len(timing_outputs) - 1,
                )
                if event is not None:
                    # if virgo is available use it for amplfi
                    if all(virgo_ready) and len(ready) == 3:
                        amplfi_ifos = ["H1", "L1", "V1"]
                    else:
                        amplfi_ifos = ["H1", "L1"]

                    process_event(
                        event_queue,
                        amplfi_queue,
                        event,
                        ifos_to_model,
                        amplfi_ifos,
                        input_buffer,
                        output_buffer,
                        samples_per_event,
                        spectral_density,
                        amplfi_whitener,
                        amplfi_psd_length,
                        shared_samples,
                        outdir,
                        device,
                    )

            # check if this is because the frame stream stopped
            # being analysis ready, in which case perform updates
            # but don't search for events
            if X is not None:
                logging.debug(
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

                state = snapshotter.reset()
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
        # so lets analyze it, taking the first two
        # channels, which correspond to H1/L1
        X = X[:2].to(device)

        # update the snapshotter state and return
        # unfolded batch of overlapping windows
        batch, state = snapshotter(X[None], state)

        # whiten the batch, and analyze with aframe
        whitened = whitener(batch)
        y = aframe(whitened)[:, 0]

        # update our output buffer with the latest aframe output,
        # which will also automatically integrate the output
        significance_outputs, timing_outputs = output_buffer.update(
            y.cpu(), t0
        )

        # if this frame was analysis ready,
        # and we had enough previous to build whitening filter
        # search for events in the integrated output
        event = None
        if snapshotter.full_psd_present and hl_ready:
            event = searcher.search(
                significance_outputs, timing_outputs, t0 + time_offset
            )

        # if we found an event, process it!
        if event is not None:
            if all(virgo_ready) and len(ready) == 3:
                amplfi_ifos = ["H1", "L1", "V1"]
            else:
                amplfi_ifos = ["H1", "L1"]
            process_event(
                event_queue,
                amplfi_queue,
                event,
                ifos_to_model,
                amplfi_ifos,
                input_buffer,
                output_buffer,
                samples_per_event,
                spectral_density,
                amplfi_whitener,
                amplfi_psd_length,
                shared_samples,
                outdir,
                device,
            )


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
    inference_params: List[str],
    channels: List[str],
    sample_rate: float,
    kernel_length: float,
    online_inference_rate: float,
    offline_inference_rate: float,
    psd_length: float,
    amplfi_psd_length: float,
    aframe_right_pad: float,
    amplfi_kernel_length: float,
    event_position: float,
    fduration: float,
    amplfi_fduration: float,
    integration_window_length: float,
    astro_event_rate: float,
    data_source: Literal["frames", "ngdd"] = "frames",
    state_channels: Optional[list[str]] = None,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    amplfi_highpass: Optional[float] = None,
    lowpass: Optional[float] = None,
    refractory_period: float = 8,
    far_threshold: float = 1,
    server: "GdbServer" = "local",
    ifo_suffix: str = None,
    input_buffer_length: int = 75,
    output_buffer_length: int = 8,
    samples_per_event: int = 20000,
    emails: Optional[list[str]] = None,
    email_far_threshold: float = 1e-6,
    auth_refresh: int = 9600,
    nside: int = 64,
    min_samples_per_pix: int = 5,
    use_distance: bool = True,
    device: str = "cpu",
    verbose: bool = False,
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
        inference_params:
            List of parameters on which the AMPLFI
            model was trained to perform inference
        channels:
            List of the channel names to analyze
        sample_rate:
            Input data sample rate in Hz
        kernel_length:
            Length of Aframe analysis kernel in seconds
        online_inference_rate:
            Rate at which to sample the output of the Aframe model,
            which determines the timing resolution for the merger
            time of detected events
        offline_inference_rate:
            Rate at which inference was performed offline when
            establishing the background and foreground distributions
        psd_length:
            Length of PSD estimation window in seconds for PSD
            used to whiten aframe data
        amplfi_psd_length:
            Length of PSD estimation window in seconds for PSD
            used to whiten amplfi data
        aframe_right_pad:
            Time offset for trigger positioning in seconds
        amplfi_kernel_length:
            Length of AMPLFI analysis window in seconds
        event_position:
            Event position (in seconds) from the left edge
            of the analysis window used for parameter estimation
        fduration:
            Length of whitening filter in seconds for aframe model
        amplfi_fduration:
            Length of whitening filter in seconds for amplfi model
        integration_window_length:
            Length of output integration window in seconds
        astro_event_rate:
            Prior on rate of astrophysical events in units Gpc^-3 yr^-1
        fftlength:
            FFT length in seconds (defaults to kernel_length + fduration)
        highpass:
            High-pass filter frequency in Hz to apply to data
            analyzed by aframe
        amplfi_highpass:
            High-pass filter frequency in Hz to apply to data
            analyzed by amplfi
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
        emails:
            List of email addresses for sending pipeline failure
            and alert emails
        email_far_threshold:
            FAR threshold in Hz at which an alert email will be sent
        auth_refresh:
            Number of seconds between calls to authenticate,
            that refreshes the scitoken credential
        min_samples_per_pix:
            Minimum number of samples per healpix pixel
            required to estimate parameters of the distance
            ansatz
        nside:
            Healpix resolution for low-latency skymaps
        use_distance:
            If true, use distance samples to create a 3D skymap
        device:
            Device to run inference on ("cpu" or "cuda")
        verbose:
            If true, autheticate with debug flag set
    """

    if emails is not None:
        logging.info(f"Sending email alerts to {', '.join(emails)}")
        send_init_email(emails, outdir)

    # validate ifos and state channels
    ifos = [channel.split(":")[0] for channel in channels]
    if ifos not in [["H1", "L1"], ["H1", "L1", "V1"]]:
        raise ValueError(
            f"Invalid interferometer configuration {ifos}. "
            "Must be ['H1', 'L1'] or ['H1', 'L1', 'V1']"
        )

    if state_channels is None:
        logging.info(
            "no state channels specified: not checking for data quality"
        )

    else:
        logging.info(
            "Checking state channels: "
            f"{', '.join(state_channels)} for data quality"
        )
        state_channels = {
            state_channel.split(":")[0]: state_channel
            for state_channel in state_channels
        }
        if set(state_channels.keys()) != set(ifos):
            raise ValueError(
                f"Specified interferometer configuration {ifos} "
                "but only specified state channels "
                f"for {list(state_channels.keys())}"
            )

    logging.info(f"{', '.join(ifos)} interferometer configuration set")

    # auth once up front before initializing gracedb client
    authenticate()
    logging.info(f"Uploading to GraceDb server: {server}")

    # Initialize GraceDB client
    gdb = gracedb_factory(
        server,
        outdir / "events",
        reload_cred=True,
        reload_buffer=MIN_VALID_LIFETIME,
    )

    fftlength = fftlength or kernel_length + fduration
    logging.info(f"Using fftlength {fftlength} for PSD estimation")

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

    subprocesses = []

    # subprocess for re-authenticating
    minsecs = MIN_VALID_LIFETIME + auth_refresh + 100
    if minsecs > SCITOKEN_LIFETIME:
        raise ValueError(
            f"Minimum requested token life {minsecs} is greater "
            f"than scitoken lifetime {SCITOKEN_LIFETIME}"
        )
    args = (error_queue, "authenticate", auth_refresh, minsecs, verbose)
    auth_process = Process(
        target=authenticate_subprocess,
        args=args,
    )
    auth_process.start()
    subprocesses.append(auth_process)

    # create subprocess for uploading initial
    # detection information like FAR to gdb
    args = (
        error_queue,
        "event creator",
        event_queue,
        gdb,
        outdir / "events",
        amplfi_queue,
        pastro_queue,
    )
    event_process = Process(
        target=event_creation_subprocess,
        args=args,
    )
    event_process.start()
    subprocesses.append(event_process)

    # initialize amplfi subprocess which
    # will recieve events via a queue
    # and process posterior samples into
    # various data products and upload them
    # to gdb

    args = (
        error_queue,
        "amplfi",
        amplfi_queue,
        gdb,
        inference_params,
        amplfi_parameter_sampler,
        shared_samples,
        emails,
        email_far_threshold,
        nside,
        min_samples_per_pix,
        use_distance,
    )

    amplfi_process = Process(
        target=amplfi_subprocess,
        args=args,
    )
    amplfi_process.start()
    subprocesses.append(amplfi_process)

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
        gdb,
        outdir,
    )
    pastro_process = Process(
        target=pastro_subprocess,
        args=args,
    )
    pastro_process.start()
    subprocesses.append(pastro_process)

    # Register cleanup function to run
    # when the main process exits
    atexit.register(cleanup_subprocesses, subprocesses)

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
        fduration=amplfi_fduration,
        amplfi_kernel_length=amplfi_kernel_length,
        event_position=event_position,
        device="cpu",
    )

    output_buffer = OutputBuffer(
        online_inference_rate=online_inference_rate,
        offline_inference_rate=offline_inference_rate,
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

    # TODO: can have something like this
    # in the config that is a mapping
    # from ifos to model weights path
    ifos_to_model = {
        ("H1", "L1", "V1"): (amplfi_hlv, scaler_hlv),
        ("H1", "L1"): (amplfi_hl, scaler_hl),
    }

    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        average="median",
    ).to(device)
    amplfi_whitener = Whiten(
        fduration=amplfi_fduration,
        sample_rate=sample_rate,
        highpass=amplfi_highpass,
        lowpass=lowpass,
    ).to(device)

    whitener = BatchWhitener(
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        inference_sampling_rate=online_inference_rate,
        batch_size=update_size * online_inference_rate,
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
        inference_sampling_rate=online_inference_rate,
    ).to(device)

    # load in background, foreground, and rejected injections;
    # use them to fit (or load in a cached) a pastro model
    logging.info("Loading background distribution")
    background = EventSet.read(background_path)
    logging.info("Background loaded")

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
        online_inference_rate=online_inference_rate,
        refractory_period=refractory_period,
        ifos=ifos,
        channels=channels,
        datadir=datadir,
        ifo_suffix=ifo_suffix,
    )

    time_offset = get_time_offset(
        online_inference_rate,
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
        ifos_to_model=ifos_to_model,
        spectral_density=spectral_density,
        amplfi_psd_length=amplfi_psd_length,
        amplfi_whitener=amplfi_whitener,
        samples_per_event=samples_per_event,
        shared_samples=shared_samples,
        data_it=data_it,
        update_size=update_size,
        time_offset=time_offset,
        device=device,
        emails=emails,
        outdir=outdir,
    )


if __name__ == "__main__":
    main()
