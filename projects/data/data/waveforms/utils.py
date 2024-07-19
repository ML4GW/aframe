import numpy as np


def convert_to_detector_frame(samples: dict[str, np.ndarray]):
    """Converts mass parameters from source to detector frame"""
    for key in ["mass_1", "mass_2", "chirp_mass", "total_mass"]:
        if key in samples:
            samples[key] = samples[key] * (1 + samples["redshift"])
    return samples
