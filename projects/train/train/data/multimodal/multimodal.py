from typing import Optional
import torch
from train.data.supervised.supervised import BaseAframeDataset
from train.data.multimodal.time_domain import TimeDomainMultimodalAframeDataset
from train.data.multimodal.frequency_domain import FrequencyDomainMultimodalAframeDataset

class MultimodalSupervisedAframeDataset(BaseAframeDataset):
    def __init__(
        self,
        *args,
        swap_prob: Optional[float] = None,
        mute_prob: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.time_dataset = TimeDomainMultimodalAframeDataset(*args, **kwargs)
        self.freq_dataset = FrequencyDomainMultimodalAframeDataset(*args, **kwargs)

    @property
    def sample_prob(self):
        return self.time_dataset.sample_prob
    
    @torch.no_grad()
    def augment(self, X, waveforms):
        strain, label, _ = self.time_dataset.augment(X, waveforms)
        (psd_low, psd_high), label_f = self.freq_dataset.augment(X, waveforms)

        assert (label == label_f).all(), "Mismatched labels between time and freq domain."

        return {
            "strain": strain,
            "psd_low": psd_low,
            "psd_high": psd_high,
            "label": label,
        }
        }
