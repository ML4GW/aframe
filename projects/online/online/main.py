import logging
from ctypes import c_wchar_p
from dataclasses import dataclass
from multiprocessing import Process, Value
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import lal
import torch
from amplfi.train.architectures.flows.base import FlowArchitecture
from architectures import Architecture
from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten

from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from online.utils.buffer import InputBuffer, OutputBuffer
from online.utils.dataloading import data_iterator
from online.utils.gdb import GdbServer, GraceDb, authenticate, gracedb_factory
from online.utils.mp import initialize_queue_processor
from online.utils.pastro import fit_or_load_pastro
from online.utils.pe import cast_samples_as_bilby_result, create_skymap
from online.utils.searcher import Event, Searcher
from online.utils.snapshotter import OnlineSnapshotter
from utils.preprocessing import BatchWhitener

if TYPE_CHECKING:
    from pastro.pastro import Pastro


# seconds of data per update
UPDATE_SIZE = 1

SECONDS_PER_DAY = 86400


def write_buffers(
    event_time: float,
    input_buffer: InputBuffer,
    output_buffer: OutputBuffer,
    outdir: Path,
):
    path = outdir / f"event_{event_time}"
    input_buffer.write(path / "strain.hdf5", event_time)
    output_buffer.write(path / "network_output.hdf5", event_time)
    logging.info("Wrote buffers to disk")


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


@dataclass
class EventProcessor:
    gdb: GraceDb
    buffer: InputBuffer
    spectral_density: SpectralDensity
    pe_whitener: Whiten
    amplfi: FlowArchitecture
    scaler: ChannelWiseScaler
    pastro_model: "Pastro"
    samples_per_event: int
    inference_params: list[str]
    outdir: Path
    nside: int
    device: str

    def submit_detection(
        self,
        event: Event,
        graceid: str,
    ):
        # write event information to disk
        # and submit it to gracedb
        event.write(self.outdir)
        logging.info("Wrote event to file")
        response = self.gdb.submit(event)
        graceid.value = response.json()["graceid"]
        # calculate and submit pastro
        logging.info("Calculating p_astro")
        pastro = self.pastro_model(event.detection_statistic)
        logging.info("Submitting p_astro")
        self.gdb.submit_pastro(float(pastro), graceid.value, event.gpstime)
        logging.info("p_astro submitted")

    def process_event(self, event: Event):
        # Define variable to be shared between parent and child process
        logging.info("Processing event")
        graceid = Value(c_wchar_p, "")
        p = Process(target=self.submit_detection, args=(event, graceid))
        p.start()

        # after event is submitted, run AMPLFI
        # to produce a posterior and skymap
        logging.info("Running AMPLFI")
        posterior, skymap, figure = self.run_amplfi(event.gpstime, self.buffer)
        logging.info("AMPLFI complete")

        # Wait for event submission to finish, and then
        # submit the posterior and skymap to gracedb
        # using the graceid from the event submission
        p.join()
        p.close()
        logging.info("Submitting PE data products")
        self.gdb.submit_pe(
            posterior, figure, skymap, graceid.value, event.gpstime
        )
        logging.info("PE submitted")

    def run_amplfi(self, event_time: float):
        # get pe data from the buffer and whiten it
        psd_strain, pe_strain = self.buffer.get_amplfi_data(event_time)
        logging.info("Got amplfi data")
        psd_strain = psd_strain.to(self.device)
        pe_strain = pe_strain.to(self.device)[None]
        logging.info("Moved amplfi data to device")
        pe_psd = self.spectral_density(psd_strain)[None]
        logging.info("Calculated amplfi PSD")
        whitened = self.amplfi_whitener(pe_strain, pe_psd)
        logging.info("Whitened amplfi data")

        # construct and highpass asd
        freqs = torch.fft.rfftfreq(
            whitened.shape[-1], d=1 / self.amplfi_whitener.sample_rate
        )
        num_freqs = len(freqs)
        pe_psd = torch.nn.functional.interpolate(
            pe_psd, size=(num_freqs,), mode="linear"
        )

        mask = freqs > self.amplfi_whitener.highpass
        pe_psd = pe_psd[:, :, mask]
        asds = torch.sqrt(pe_psd)
        logging.info("Calculated ASD")

        # sample from the model and descale back to physical units
        samples = self.amplfi.sample(
            self.samples_per_event, context=(whitened, asds)
        )
        descaled_samples = self.std_scaler(samples.mT, reverse=True).mT.cpu()
        logging.info("Sampled and scaled")

        indices = [
            self.inference_params.index(p)
            for p in ["chirp_mass", "mass_ratio", "distance"]
        ]
        posterior = cast_samples_as_bilby_result(
            descaled_samples[..., indices].numpy(),
            ["chirp_mass", "mass_ratio", "distance"],
            f"{event_time} result",
        )
        logging.info("Cast samples")

        phi_idx = self.inference_params.index("phi")
        dec_idx = self.inference_params.index("dec")
        ra = (
            torch.remainder(
                lal.GreenwichMeanSiderealTime(event_time)
                + descaled_samples[..., phi_idx],
                torch.as_tensor(2 * torch.pi),
            )
            - torch.pi
        )
        dec = descaled_samples[..., dec_idx] + torch.pi / 2

        skymap, figure = create_skymap(
            ra,
            dec,
            self.nside,
            title=f"{event_time} sky map",
        )
        logging.info("Created skymap")

        return posterior, skymap, figure


@torch.no_grad()
def search(
    snapshotter: OnlineSnapshotter,
    whitener: Whiten,
    searcher: Searcher,
    event_processor: EventProcessor,
    output_buffer: OutputBuffer,
    write_buffers: bool,
    aframe: Architecture,
    data_it: Iterable[Tuple[torch.Tensor, float, bool]],
    time_offset: float,
    outdir: Path,
    device: str,
):
    integrated = None

    # flag that declares if the most previous frame
    # was analysis ready or not
    in_spec = False

    # Set up a queue of events to be processed and written
    event_queue = initialize_queue_processor(event_processor.process_event)
    if write_buffers:
        buffer_write_queue = initialize_queue_processor(write_buffers)

    # Set up variables for writing buffers to disk
    last_event_written = True
    last_event_time = 0

    state = snapshotter.initial_state
    for X, t0, ready in data_it:
        # if this frame was not analysis ready
        if not ready:
            if searcher.detecting:
                # if we were in the middle of a detection,
                # we won't get to see the peak of the event
                # so build the event with what we have
                event = searcher.build_event(
                    integrated[-1], t0 - 1, len(integrated) - 1
                )
                if event is not None:
                    # maybe process event found in the previous frame
                    event_queue.put((event,))
                    last_event_written = False
                    last_event_time = event.gpstime
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

                event_processor.buffer.reset()
                output_buffer.reset()

                # nothing left to do, so move on to next frame
                continue

        elif not in_spec:
            # the frame is analysis ready, but previous frames
            # weren't, so reset our running states
            logging.info(f"Frame {t0} is ready again, resetting states")
            state = snapshotter.reset()
            event_processor.buffer.reset()
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
        event_processor.buffer.update(X.cpu(), t0)
        # update our output buffer with the latest aframe output,
        # which will also automatically integrate the output
        integrated = output_buffer.update(y.cpu(), t0)

        # if this frame was analysis ready,
        # and we had enough previous to build whitening filter
        # search for events in the integrated output
        event = None
        if snapshotter.full_psd_present and ready:
            event = searcher.search(integrated, t0 + time_offset)

        # if we found an event, add it to the processing queue!
        if event is not None:
            event_queue.put((event,))
            searcher.detecting = False
            last_event_written = False
            last_event_time = event.gpstime

        # write buffers to disk, waiting a little while after the event
        # to get a more complete picture
        if (
            write_buffers
            and not last_event_written
            and t0 > last_event_time + output_buffer.buffer_length / 2
        ):
            buffer_write_queue.put(
                (
                    last_event_time,
                    event_processor.buffer,
                    output_buffer,
                    outdir,
                )
            )


def main(
    aframe_weights: Path,
    amplfi_architecture: FlowArchitecture,
    amplfi_weights: Path,
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
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    refractory_period: float = 8,
    far_threshold: float = 1,
    server: GdbServer = "test",
    ifo_suffix: str = None,
    input_buffer_length: int = 75,
    output_buffer_length: int = 8,
    write_buffers: bool = False,
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
        write_buffers:
            Flag that determines whether to write buffer contents
            to disk after an event
        samples_per_event:
            Number of posterior samples to generate per event
            for creating skymaps and other parameter estimation
            data products
        nside:
            Healpix resolution for low-latency skymaps
        device:
            Device to run inference on ("cpu" or "cuda")
    """
    # run htgettoken and kinit
    if server != "local":
        authenticate()

    gdb = gracedb_factory(server, outdir)

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

    # Amplfi setup
    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        average="median",
    ).to(device)
    pe_whitener = Whiten(
        fduration=fduration, sample_rate=sample_rate, highpass=highpass
    ).to(device)

    # load in background, foreground, and rejected injections;
    # use them to fit (or load in a cached) a pastro model
    background = EventSet.read(background_path)
    foreground = RecoveredInjectionSet.read(foreground_path)
    rejected = InjectionParameterSet.read(rejected_path)

    # if event set is not sorted by detection statistic, sort it
    # which will significantly speed up the far calculation
    if not background.is_sorted_by("detection_statistic"):
        logging.info(
            "Sorting background by detection statistic and writing to disk"
        )
        background = background.sort_by("detection_statistic")
        background.write(background_path)

    pastro_model = fit_or_load_pastro(
        outdir / "pastro.pkl",
        background,
        foreground,
        rejected,
        astro_event_rate=astro_event_rate,
    )

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

    data_it = data_iterator(
        datadir=datadir,
        channels=channels,
        ifos=ifos,
        sample_rate=sample_rate,
        ifo_suffix=ifo_suffix,
        timeout=10,
    )

    event_processor = EventProcessor(
        gdb=gdb,
        buffer=input_buffer,
        spectral_density=spectral_density,
        pe_whitener=pe_whitener,
        amplfi=amplfi,
        scaler=scaler,
        pastro_model=pastro_model,
        samples_per_event=samples_per_event,
        inference_params=inference_params,
        outdir=outdir,
        nside=nside,
        device=device,
    )

    search(
        snapshotter=snapshotter,
        whitener=whitener,
        searcher=searcher,
        event_processor=event_processor,
        output_buffer=output_buffer,
        write_buffers=write_buffers,
        aframe=aframe,
        data_it=data_it,
        time_offset=time_offset,
        outdir=outdir,
        device=device,
    )


if __name__ == "__main__":
    main()
