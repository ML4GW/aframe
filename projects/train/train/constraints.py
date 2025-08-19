import torch


def mass_ratio_constraint(
    parameters: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Enforce mass_2 < mass_1. Assumes that the given parameter dictionary
    already contains the component masses
    """
    try:
        mask = parameters["mass_1"] > parameters["mass_2"]
    except KeyError as exc:
        raise ValueError(
            "Parameter dictionary did not contain component masses"
        ) from exc

    return {key: parameters[key][mask] for key in parameters.keys()}
