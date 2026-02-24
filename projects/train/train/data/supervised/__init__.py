from .multimodal import MultiModalSupervisedAframeDataset
from .supervised import SupervisedAframeDataset
from .time_domain import (
    HeterodyneTimeDomainSupervisedAframeDataset,
    TimeDomainSupervisedAframeDataset,
    TimeDomainSupervisedRegressionDataset,
)
from .time_frequency_domain import (
    FrequencyDomainSupervisedAframeDataset,
    SpectrogramDomainSupervisedAframeDataset,
    TimeSpectrogramDomainSupervisedAframeDataset,
)
