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

import torch
from utils.preprocessing import BackgroundSnapshotter, BatchWhitener


def build_model(weights, backend, device, aoti_path=None):
    """Load the trained NN by backend."""
    if backend == "aoti":
        if aoti_path is None:
            raise ValueError("backend 'aoti' requires aoti_path")
        # aoti_load_package references torch._inductor.codecache without
        # importing it
        import torch._inductor.codecache  # noqa: F401

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
        self._result = None
        logging.info(f"LocalInferenceClient ready (backend={backend})")

    def reset(self):
        """Clear between branches when the client is reused"""
        self._states.clear()
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

    def get(self, until_empty: bool = False):
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._states.clear()
        return False
