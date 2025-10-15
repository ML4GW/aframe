from typing import Callable, Optional

import torch


class AframePrior:
    def __init__(
        self,
        priors: dict[str, torch.distributions.Distribution],
        conversion_function: Optional[Callable] = None,
        constraint_function: Optional[Callable] = None,
    ):
        """
        A class for sampling parameters from a prior distribution

        Args:
            priors:
                A dictionary of parameter samplers that take an integer N
                and return a tensor of shape (N, ...) representing
                samples from the prior distribution
            conversion_function:
                A callable that takes a dictionary of sampled parameters
                and returns a dictionary of waveform generation parameters
            constraint_function:
                A callable that takes a dictionary of sampled parameters,
                discards any that don't satisfy the constraint, and returns
                the remaining parameters
        """
        super().__init__()
        self.priors = priors
        self.conversion_function = conversion_function or (lambda x: x)
        self.constraint_function = constraint_function or (lambda x: x)

    def __call__(
        self,
        N: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """
        Generates random samples from the prior

        Args:
            N: Number of samples to generate
            device: Device to place the samples
        """
        # sample parameters from prior
        parameters = {
            k: v.sample((N,)).to(device) for k, v in self.priors.items()
        }
        # perform any necessary conversions
        # to from sampled parameters to
        # waveform generation parameters
        parameters = self.conversion_function(parameters)

        # Discard any samples that don't meet the constraint
        parameters = self.constraint_function(parameters)

        keys = list(parameters.keys())
        while len(parameters[keys[0]]) < N:
            new_params = {
                k: v.sample((N,)).to(device) for k, v in self.priors.items()
            }
            new_params = self.conversion_function(parameters)
            new_params = self.constraint_function(parameters)
            parameters = {
                key: torch.cat([parameters[key], new_params[key]])
                for key in keys
            }

        return {k: v[:N] for k, v in parameters.items()}

    def log_prob(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the log probability of samples under the prior.
        TODO: This calculation is incorrect if a constraint function
        is specified. We don't use this function anywhere currently,
        so not too big a deal, but we'll need to fix this if we do.

        Args:
            samples:
                Dictionary where key is parameter and
                value is tensor of samples
        """

        first = samples[list(samples.keys())[0]]
        log_probs = torch.ones(len(first), device=first.device)
        for param, tensor in samples.items():
            log_probs += self.priors[param].log_prob(tensor).to(first.device)
        return log_probs


class ParameterTransformer(torch.nn.Module):
    """
    Helper class for applying preprocessing
    transformations to inference parameters
    """

    def __init__(self, **transforms: Callable):
        super().__init__()
        self.transforms = transforms

    def forward(
        self,
        parameters: dict[str, torch.Tensor],
    ):
        # transform parameters
        transformed = {k: v(parameters[k]) for k, v in self.transforms.items()}
        # update parameter dict
        parameters.update(transformed)
        return parameters
