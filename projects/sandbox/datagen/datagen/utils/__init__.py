from typing import List, Optional


def get_state_flags(ifos: List[str], state_flag: str):
    if state_flag == "DATA":
        flags = [f"{ifo}_DATA" for ifo in ifos]  # open data flags
    else:
        flags = [f"{ifo}:{state_flag}" for ifo in ifos]
    return flags


def get_channels(ifos: List[str], channel: str):
    if channel == "OPEN":
        channels = ifos
    else:
        channels = [f"{ifo}:{channel}" for ifo in ifos]
    return channels
