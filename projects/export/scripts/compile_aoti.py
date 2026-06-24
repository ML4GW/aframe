# ruff: noqa: F821
"""AOTInductor-compile the trained model.

The in-process analogue of the TensorRT export step for the `aoti` inference
backend. Loads model_exported.pt2 and AOTInductor-compiles it at the fixed
inference batch shape. The resulting package is GPU-specific, so inference
must be done on the same GPU as compilation.
"""

import torch

params = snakemake.params
out = snakemake.output[0]
num_samples = int(float(params.kernel_length) * float(params.sample_rate))

model = torch.export.load(snakemake.input.exported).module().to("cuda")
example = (
    torch.randn(
        int(params.batch_size),
        int(params.num_ifos),
        num_samples,
        device="cuda",
    ),
)
# The model was originally exported on CPU, so re-export on
# the current device to match the data.
# torch 2.12+ replaces this with torch.compile(fullgraph=True).aot_compile()
exported_program = torch.export.export(model, example)
torch._inductor.aoti_compile_and_package(exported_program, package_path=out)
