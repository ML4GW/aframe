from .base import Architecture
from .supervised import (
    SupervisedArchitecture,
    SupervisedFrequencyDomainResNet,
    SupervisedSpectrogramDomainResNet,
    SupervisedTimeDomainResNet,
)
from .multimodal import (
        MultimodalSupervisedArchitecture
)
from .concat import ConcatResNet
