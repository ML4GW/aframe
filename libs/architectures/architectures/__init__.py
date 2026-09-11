from .base import Architecture
from .regression import (
    MultiTaskArchitecture,
    MultiTaskTimeDomainResNet,
    RegressionArchitecture,
    RegressionTimeDomainResNet,
)
from .supervised import (
    SupervisedArchitecture,
    SupervisedFrequencyDomainResNet,
    SupervisedMultiModalResNet,
    SupervisedSpectrogramDomainResNet,
    SupervisedTimeDomainResNet,
    SupervisedTimeSpectrogramResNet,
    SupervisedHeterodyneTimeDomainResNet,
)
