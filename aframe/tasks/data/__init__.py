from .fetch import Fetch
from .segments import Query
from .waveforms import TestingWaveforms, TrainingWaveforms, ValidationWaveforms

DATAFIND_ENV_VARS = [
    "KRB5_KTNAME",
    "X509_USER_PROXY",
    "GWDATAFIND_SERVER",
    "NDSSERVER",
    "LIGO_USERNAME",
    "DEFAULT_SEGMENT_SERVER",
]
