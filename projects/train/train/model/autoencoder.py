import torch
from architectures.autoencoder import AutoencoderArchitecture
from ml4gw.transforms import ShiftedPearsonCorrelation

from train.model.base import AframeBase

Tensor = torch.Tensor


class AutoencoderAframe(AframeBase):
    # TODO: include extra init arguments for the
    # various loss terms we might like to include.
    # If specific architectures have loss functions
    # that need to be included, e.g. beta VAEs, include
    # them as a `loss` attribute on the architecture
    # that gets set during its forward pass.
    def __init__(
        self,
        arch: AutoencoderArchitecture,
        max_shift: float = 0.005,
        *args,
        **kwargs
    ) -> None:
        super().__init__(arch, *args, **kwargs)

    def on_validation_start(self):
        """
        Override this because on_train_start happens
        after sanity checking, so in case we do a
        sanity check we need to instantiate the
        pearson module.
        """
        if self.trainer.sanity_checking:
            self.on_train_start()

    def on_train_start(self):
        """
        Build our Pearson correlation here since
        we won't know the sample rate up front.
        """
        sample_rate = self.trainer.datamodule.sample_rate
        shift = int(self.hparams.max_shift * sample_rate)
        self.pearson = ShiftedPearsonCorrelation(shift)
        self.pearson.to(self.device)

    def forward(self, X):
        X_hat = self.model(X)

        # Reverse X_hat along the channel dimension
        # to indicate that each inteferometer is
        # producing a prediction for the other.
        # TODO: this will need to be more complicated for
        # when there's more than one interferometer
        return torch.flip(X_hat, dims=(1,))

    def get_losses(self, X, X_hat):
        # this will have shape (shifts, batch, ifos)
        correlation = self.pearson(X, X_hat)

        # stupidest way to do this: just take
        # the max pearson correlation across
        # all shifts (0th dimension).
        # TODO: potentially better ideas:
        # 1) Enforce some prior on the expected
        #    shape of the correlation around its max value
        # 2) Implement the chi-squared metric using X_hat
        #    and X (you don't need the PSD because X has
        #    already divided by it during whitening). Then
        #    include an extra loss term that penalizes
        #    chi-squared values greater than 1. One way
        #    might be to downweight the correlation values
        #    by the chi-squared value using eq. 18 fromm
        #    this paper https://arxiv.org/abs/1208.3491
        # 3) Imposing a penalty on higher-frequency components
        #    in X_hat to keep it from getting lucky on
        #    on one of the 80 or so shifts it gets to try hitting
        correlation = correlation.max(dim=0).values

        # combine the correlation in quadrature across the IFOS
        correlation = (correlation**2).sum(-1) ** 0.5

        # The best way to implement a more complicated loss
        # scheme would be to return each of the terms separately
        # in a dictionary and return, then combine them using
        # self.compute_loss_fn.
        return {"correlation": correlation}

    def train_step(self, X):
        X_hat = self(X)
        return self.get_losses(X, X_hat)

    def compute_loss_fn(self, correlation):
        # this is a simple dummy version of this, include
        # more arguments and return them during train_step
        # to do more complex stuff. Note that we'll return
        # the negative since gradient descent will attempt
        # to minimize the loss
        return -correlation

    def score(self, X):
        """
        Create a detection statistic by taking the
        negative of the loss, since low loss values
        correspond to high correlation aka to likely events.
        """

        X_hat = self(X)
        if self.validating:
            self.on_validation_score(X, X_hat)
        losses = self.get_losses(X, X_hat)
        return -self.compute_loss_fn(**losses)
