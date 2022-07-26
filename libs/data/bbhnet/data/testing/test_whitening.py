from pathlib import Path

import h5py
import torch

from bbhnet.data.transforms import WhiteningTransform

sample_rate = 2048
kernel_length = 1

data_path = Path("/home/ethan.marx/bbhnet/first_training_run/data/")


with h5py.File(data_path / "H1_background.h5") as f:
    hanford_background = f["hoft"][:]


with h5py.File(data_path / "glitches.h5") as f:
    glitches = f["H1_glitches"][()]

num_ifos = 1

background = torch.Tensor(hanford_background)[None, :]

preprocessor = WhiteningTransform(num_ifos, sample_rate, kernel_length)
preprocessor.fit(background)
