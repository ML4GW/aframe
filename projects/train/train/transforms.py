import torch

Tensor = torch.Tensor


class ChirpMass:
    """Adds ``chirp_mass`` computed from ``mass_1`` and ``mass_2``."""

    def __call__(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        m1, m2 = params["mass_1"], params["mass_2"]
        params["chirp_mass"] = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        return params


class MassRatio:
    """Adds ``mass_ratio`` = min(m1,m2) / max(m1,m2) in range (0, 1]."""

    def __call__(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        m1, m2 = params["mass_1"], params["mass_2"]
        params["mass_ratio"] = torch.minimum(m1, m2) / torch.maximum(m1, m2)
        return params
