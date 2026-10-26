"""In-process inference setup that mirrors the Hermes InferenceClient.

Runs the snapshotter -> whitener -> NN ensemble locally on the GPU instead
of streaming to a Triton server. Has the same interface as InferenceClient,
so it can be used in main.infer() with the same Sequence and Postprocessor.

Supports three backends:
    export: load ExportedProgram and run as a GraphModule. GPU agnostic
    compile: load ExportedProgram + torch.compile. GPU agnostic, per-job warmup
    aoti: load pre-compiled AOTInductor package. GPU specific, no warmup.
"""

import logging

import numpy as np
import torch
from ml4gw.utils.slicing import unfold_windows
from utils.preprocessing import BackgroundSnapshotter, BatchWhitener


def build_model(weights, backend, device, aoti_path=None):
    """Load the trained NN by backend."""
    if backend == "aoti":
        if aoti_path is None:
            raise ValueError("backend 'aoti' requires aoti_path")
        # aoti_load_package references torch._inductor.codecache without
        # importing it. `import torch._inductor.codecache` would make
        # `torch` local to this function, breaking the other backends.
        from torch._inductor import codecache  # noqa: F401

        runner = torch._inductor.aoti_load_package(str(aoti_path))

        def model(kernels):
            return runner(kernels)[0]

        return model

    if backend in ("export", "compile"):
        exported = torch.export.load(weights)
        module = exported.module().to(device)
        return torch.compile(module) if backend == "compile" else module
    raise ValueError(
        f"backend must be 'export', 'compile', or 'aoti', got {backend}"
    )


class LocalInferenceClient:
    """Run the streaming ensemble locally. Same interface as InferenceClient"""

    def __init__(
        self,
        weights: str,
        backend: str = "export",
        aoti_path: str | None = None,
        num_ifos: int = 2,
        psd_length: float = 64.0,
        kernel_length: float = 1.5,
        fduration: float = 1.0,
        sample_rate: float = 2048.0,
        inference_sampling_rate: float = 4.0,
        batch_size: int = 128,
        highpass: float = 32.0,
        fftlength: float | None = None,
        device: str = "cuda",
        callback=None,
    ):
        self.device = device
        self.callback = callback
        self.num_ifos = num_ifos
        self.model = build_model(weights, backend, device, aoti_path)
        self.snapshotter = BackgroundSnapshotter(
            psd_length=psd_length,
            kernel_length=kernel_length,
            fduration=fduration,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
        ).to(device)
        self.whitener = BatchWhitener(
            kernel_length=kernel_length,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            batch_size=batch_size,
            fduration=fduration,
            fftlength=fftlength,
            highpass=highpass,
        ).to(device)
        # one streaming snapshot state per sequence id (background + injection)
        self._states: dict[int, torch.Tensor] = {}
        self._stream_state: torch.Tensor | None = None
        self._result = None
        logging.info(f"LocalInferenceClient ready (backend={backend})")

    def reset(self):
        """Clear between branches when the client is reused"""
        self._states.clear()
        self._stream_state = None
        self._result = None

    def _state(self, sequence_id):
        if sequence_id not in self._states:
            self._states[sequence_id] = torch.zeros(
                2,  # background + injection
                self.num_ifos,
                self.snapshotter.state_size,
                device=self.device,
            )
        return self._states[sequence_id]

    def infer(
        self,
        x,
        request_id=None,
        sequence_id=None,
        sequence_start=False,
        sequence_end=False,
    ):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=self.device)
            full, state = self.snapshotter(xt, self._state(sequence_id))
            self._states[sequence_id] = state
            y = self.model(self.whitener(full))
        result = self.callback(
            y.detach().cpu().numpy(), request_id, sequence_id
        )
        if result is not None:
            self._result = result
        if sequence_end:
            del self._states[sequence_id]

    def infer_streams(
        self, update: np.ndarray, psd_source: list[int]
    ) -> np.ndarray:
        """
        Send a batch update with several streams through the model.

        Args:
            update:
                The new samples for this step, of shape `(N, C, step)`.
            psd_source:
                `N` indices. The `i`th stream of `update` is whitened by
                the PSD of stream `psd_source[i]`.

        Returns:
            Array of model outputs with shape `(N, batch_size)`.

        """
        num_streams, num_channels, _ = update.shape

        with torch.no_grad():
            # Move the new data to the device and append it to the streams'
            # snapshot state.
            x = torch.tensor(update, dtype=torch.float32, device=self.device)
            if self._stream_state is None:
                self._stream_state = torch.zeros(
                    num_streams,
                    num_channels,
                    self.snapshotter.state_size,
                    device=self.device,
                )
            full, state = self.snapshotter(x, self._stream_state)
            self._stream_state = state

            # Do the splitting, PSD estimation, and whitening manually, as
            # opposed to using the BatchWhitener, to avoid the logic in the
            # PsdEstimator that says that a batch of exactly two means that
            # the data from the second batch element should be whitened with
            # the PSD from the first batch element.
            size = self.whitener.psd_estimator.size
            psd_part, data = torch.split(
                full, [full.shape[-1] - size, size], dim=-1
            )
            psds = self.whitener.psd_estimator.spectral_density(
                psd_part[psd_source].double()
            )
            whitened = self.whitener.whitener(data.double(), psds)

            # Unfold the whitened streams into kernels, and reshape so
            # that the ordering goes stream by stream and different streams'
            # outputs aren't interleaved.
            kernels = unfold_windows(
                whitened,
                self.whitener.kernel_size,
                self.whitener.stride_size,
            ).transpose(0, 1)

            kernels = kernels.reshape(
                -1, num_channels, self.whitener.kernel_size
            )

            y = self.model(kernels)
        return y.cpu().reshape(num_streams, -1).numpy()

    def get(self, until_empty: bool = False):
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._states.clear()
        return False
