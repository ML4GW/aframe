import io
import logging
from typing import Optional

import h5py
import hermes.quiver as qv
import torch
from utils.s3 import open_file

from architectures.stackedresnets import MultimodalMultiband
from ml4gw.nn.resnet.resnet_1d import GroupNorm1DGetter
from collections.abc import Sequence
import os

Tensor = torch.Tensor
def separate_model(weights: str,
                   batch_file: str,
                   num_ifos: int,
                   kernel_length: float,
                   sample_rate: float,
                   batch_size: int,
                   classes: tuple,
                   layers: tuple,
                   inference_sampling_rates: tuple,
                   **kwargs,):
    classes = list(classes)
    inference_sampling_rates = list(inference_sampling_rates)
    layers = [list(v) for v in layers]
    mm_test=MultimodalMultiband(classes = classes, num_ifos = num_ifos, sample_rate = sample_rate,  kernel_length = kernel_length, 
                                layers = layers, kernel_size = 3, zero_init_residual = False, groups = 1, width_per_group = 64, 
                                stride_type = None, norm_layer = GroupNorm1DGetter(groups = 16))
    
    with open_file(weights, "rb") as f:
        graph = nn = torch.jit.load(f, map_location="cpu")

    graph.eval()
    with open_file(batch_file, "rb") as f:
        batch_file = h5py.File(io.BytesIO(f.read()))
    
    layers = sorted(batch_file.keys() - "y")
    input_shapes = [(batch_size*inference_sampling_rates[i]//max(inference_sampling_rates), 
                     batch_file[layer].shape[-2], 
                     batch_file[layer].shape[-1]) for i, layer in enumerate(layers)]
    
    mm_test.load_state_dict(nn.state_dict())
    model_parent_dir = os.path.dirname(weights)
    for i in range(len(layers)):
        submodule = mm_test.get_submodule(f'resnets.{i}')
        test_input=torch.ones(input_shapes[i])
        net_trace = torch.jit.trace(submodule, test_input)
        outpath = os.path.join(model_parent_dir, f'resnets_{i}.pt')
        torch.jit.save(net_trace, outpath)
    submodule = mm_test.get_submodule('fc')
    test_input=torch.ones((batch_size, sum(classes)))
    net_trace = torch.jit.trace(submodule, test_input)
    outpath = os.path.join(model_parent_dir, 'fc.pt')
    torch.jit.save(net_trace, outpath)
    return

class concatenation_layer(torch.nn.Module): #the concatenation is made with explicit args so that triton can keep track of variables
    def __init__(self,
                 inference_sampling_rates: tuple,
                ) -> None:
        super().__init__()
        inference_sampling_rates = list(inference_sampling_rates)
        self.repeats = [max(inference_sampling_rates)//x for x in inference_sampling_rates]
        self.num_layers = len(self.repeats)
        self.forward = make_forward(self.num_layers).__get__(self)

def make_forward(num_layers):
    args = ", ".join(f"classes_{i}: Tensor" for i in range(num_layers))
    inputs = ", ".join(f"classes_{i}" for i in range(num_layers))
    code = f"""
def forward(self, {args}):
    x = ({inputs})
    x = tuple(torch.repeat_interleave(x[i], self.repeats[i], dim = 0) for i in range(self.num_layers))
    x = torch.cat(x, 1).flatten(1)
    return x
"""
    local_vars = {}
    exec(code, globals(), local_vars)
    return local_vars['forward']