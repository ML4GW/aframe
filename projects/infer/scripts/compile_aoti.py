# ruff: noqa: F821
"""AOTInductor-compile the trained model.

The in-process analogue of the TensorRT export step for the `aoti` inference
backend. Loads model_exported.pt2 and AOTInductor-compiles it for any number
of kernels per call. The resulting package is GPU-specific, so inference
must be done on the same GPU as compilation.
"""

from infer.local import compile_aoti

params = snakemake.params
compile_aoti(
    snakemake.input.exported,
    snakemake.output[0],
    batch_size=int(params.batch_size),
    num_ifos=int(params.num_ifos),
    num_samples=int(float(params.kernel_length) * float(params.sample_rate)),
    device="cuda",
)
