import numpy as np
import torch
from gwpy.timeseries import TimeSeries

from bbhnet.data.transforms.whitening import (
    DEFAULT_FFTLENGTH,
    WhiteningTransform,
)


def test_whitening_transform(data_length, sample_rate, num_ifos):
    whitener = WhiteningTransform(num_ifos, sample_rate, 1)
    assert len(list(whitener.parameters())) == 1
    assert whitener.time_domain_filter.ndim == 3
    assert len(whitener.time_domain_filter) == num_ifos
    assert whitener.time_domain_filter.shape[-1] == (sample_rate - 1)
    assert len(whitener.window) == sample_rate

    background = torch.randn(num_ifos, int(data_length * sample_rate))
    whitener.fit(background)

    raw = torch.randn(8, num_ifos, sample_rate)
    whitened = whitener(raw)

    # now do everything with gwpy and make sure they align
    asds = []
    for ifo in background.cpu().numpy():
        ts = TimeSeries(ifo, dt=1 / sample_rate)
        asd = ts.asd(
            fftlength=DEFAULT_FFTLENGTH, method="median", window="hanning"
        )
        asds.append(asd)

    for x, y in zip(raw.cpu().numpy(), whitened.cpu().numpy()):
        for ifo, output, asd in zip(x, y, asds):
            ts = TimeSeries(ifo, dt=1 / sample_rate)
            target = ts.whiten(
                fflength=DEFAULT_FFTLENGTH,
                method="median",
                window="hanning",
                asd=asd,
            ).value

            # TODO: this agreement isn't very tight but I can't
            # get it consistently closer to this. Is this just
            # numerical precision issues adding up?
            assert np.isclose(target, output, rtol=1e-3).all()

    # TODO: test shape checks on fit and forward
