import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from online_deployment.buffer import DataBuffer
from online_deployment.dataloading import data_iterator
from online_deployment.snapshot_whitener import SnapshotWhitener
from online_deployment.trigger import Searcher, Trigger

from aframe.architectures import architecturize
from utils.logging import configure_logging


@architecturize
@torch.no_grad()
def main(
    architecture: Callable,
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
    far_per_day: float = 1,
    secondary_far_threshold: float = 24,
    server: str = "test",
    ifo_suffix: str = None,
    input_buffer_length=75,
    output_buffer_length=8,
    verbose: bool = False,
):
    logdir = outdir / "log"
    logdir.mkdir(exist_ok=True, parents=True)
    # Create a new log file each time we start using the current UTC time
    log_suffix = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    configure_logging(outdir / "log" / f"deploy_{log_suffix}.log", verbose)

    num_ifos = len(ifos)
    buffer = DataBuffer(
        num_ifos,
        sample_rate,
        inference_sampling_rate,
        integration_window_length,
        input_buffer_length,
        output_buffer_length,
    )

    # instantiate network and load in its optimized parameters
    weights_path = outdir / "training" / "weights.pt"
    logging.info(f"Build network and loading weights from {weights_path}")

    nn = architecture(num_ifos).to("cuda")
    fftlength = fftlength or kernel_length + fduration
    whitener = SnapshotWhitener(
        num_channels=num_ifos,
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
        fftlength=fftlength,
        highpass=highpass,
    )
    current_state = whitener.get_initial_state().to("cuda")

    weights = torch.load(weights_path)
    nn.load_state_dict(weights)
    nn.eval()

    # set up some objects to use for finding
    # and submitting triggers
    fars = [far_per_day, secondary_far_threshold]
    searcher = Searcher(
        outdir, fars, inference_sampling_rate, refractory_period
    )

    triggers = [
        Trigger(server=server, write_dir=outdir / server / "triggers"),
        Trigger(
            server=server, write_dir=outdir / server / "secondary-triggers"
        ),
    ]
    in_spec = True

    def get_trigger(event):
        fars_hz = [i / 3600 / 24 for i in fars]
        idx = np.digitize(event.far, fars_hz)
        if idx == 0 and not in_spec:
            logging.warning(
                "Not submitting event {} to production trigger "
                "because data is not analysis ready".format(event)
            )
            idx = 1
        return triggers[idx]

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

    logging.info("Beginning search")
    data_it = data_iterator(
        datadir=datadir,
        channel=channel,
        ifos=ifos,
        sample_rate=sample_rate,
        ifo_suffix=ifo_suffix,
        timeout=10,
    )
    integrated = None  # need this for static linters
    last_event_written = True
    last_event_time = 0
    for X, t0, ready in data_it:
        # adjust t0 to represent the timestamp of the
        # leading edge of the input to the network
        if not ready:
            in_spec = False

            # if we had an event in the last frame, we
            # won't get to see its peak, so do our best
            # to build the event with what we have
            if searcher.detecting and not searcher.check_refractory:
                event = searcher.build_event(
                    integrated[-1], t0 - 1, len(integrated) - 1
                )
                trigger = get_trigger(event)
                trigger.submit(event, ifos, datadir, ifo_suffix)
                searcher.detecting = False
                last_event_written = False
                last_event_trigger = trigger
                last_event_time = event.gpstime

            # check if this is because the frame stream stopped
            # being analysis ready, or if it's because frames
            # were dropped within the stream
            if X is not None:
                logging.warning(
                    "Frame {} is not analysis ready. Performing "
                    "inference but ignoring any triggers".format(t0)
                )
            else:
                logging.warning(
                    "Missing frame files after timestep {}, "
                    "resetting states".format(t0)
                )
                # Write whatever data we have from the event
                if not last_event_written:
                    write_path = last_event_trigger.write_dir
                    buffer.write(write_path, last_event_time)
                    last_event_written = True
                buffer.reset_state()
                continue
        elif not in_spec:
            # the frame is analysis ready, but previous frames
            # weren't, so reset our running states
            logging.info(f"Frame {t0} is ready again, resetting states")
            current_state = whitener.get_initial_state().to("cuda")
            buffer.reset_state()
            in_spec = True

        X = X.to("cuda")
        batch, current_state, full_psd_present = whitener(X, current_state)
        y = nn(batch)[:, 0]
        integrated = buffer.update(
            input_update=X,
            output_update=y,
            t0=t0,
            input_time_offset=0,
            output_time_offset=time_offset + integration_window_length,
        )

        event = None
        # Only search if we had sufficient data to whiten with
        # and if frames were analysis ready
        if full_psd_present and ready:
            event = searcher.search(integrated, t0 + time_offset)

        if event is not None:
            trigger = get_trigger(event)
            trigger.submit(event, ifos, datadir, ifo_suffix)
            last_event_written = False
            last_event_trigger = trigger
            last_event_time = event.gpstime

        if (
            not last_event_written
            and last_event_time + output_buffer_length / 2 < t0
        ):
            write_path = last_event_trigger.write_dir
            buffer.write(write_path, last_event_time)
            last_event_written = True
