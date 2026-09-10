import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.config.paths import get_cache_dir
from ledger.events import veto_mask

from plots.vetos import (
    GATE_PATHS,
    VETO_CATEGORIES,
    VETO_DEFINER_FILE,
    VetoParser,
    get_catalog_vetos,
)

if TYPE_CHECKING:
    from ledger.events import EventSet

DEFAULT_SEGMENTS_CACHE = Path(get_cache_dir("aframe")) / "veto_segments.json"


def combine_masks(
    masks: dict[str, np.ndarray], selected: Sequence[str]
) -> np.ndarray:
    """OR the masks for `selected` categories into one veto mask."""
    n = len(next(iter(masks.values()))) if masks else 0
    combined = np.zeros(n, dtype=bool)
    for cat in selected:
        if cat not in masks:
            raise KeyError(
                f"Unknown veto category {cat!r}, expected one of {list(masks)}"
            )
        combined |= masks[cat]
    return combined


def _segments_key(
    categories: Sequence[str], ifos: Sequence[str], start: float, stop: float
) -> str:
    """Cache key covering both the query and the source files it reads."""
    h = hashlib.sha256()
    for path in [VETO_DEFINER_FILE, *GATE_PATHS.values()]:
        if path.exists():
            h.update(path.read_bytes())
    query = {
        "categories": sorted(categories),
        "ifos": sorted(ifos),
        "start": start,
        "stop": stop,
    }
    h.update(json.dumps(query, sort_keys=True).encode())
    return h.hexdigest()


def load_or_fetch_segments(
    categories: Sequence[VETO_CATEGORIES],
    ifos: Sequence[str],
    start: float,
    stop: float,
    cache: Path = DEFAULT_SEGMENTS_CACHE,
) -> dict[str, dict[str, np.ndarray]]:
    """Segment lookup for `categories`, cached to `cache` on disk.

    Returns:
        `{category: {ifo: (N, 2) array of [start, end) segment bounds}}`.
    """
    key = _segments_key(categories, ifos, start, stop)
    if cache.exists():
        with open(cache) as f:
            cached = json.load(f)
        if cached.get("key") == key:
            logging.info(f"Using cached veto segments from {cache}")
            return {
                cat: {ifo: np.asarray(segs) for ifo, segs in ifo_segs.items()}
                for cat, ifo_segs in cached["segments"].items()
            }

    segments: dict[str, dict[str, np.ndarray]] = {}
    parser_categories = [c for c in categories if c != "CATALOG"]
    if parser_categories:
        veto_parser = VetoParser(
            VETO_DEFINER_FILE, GATE_PATHS, start, stop, ifos
        )
        for cat in parser_categories:
            segments[cat] = veto_parser.get_vetos(cat)
    if "CATALOG" in categories:
        catalog = get_catalog_vetos(start, stop)
        segments["CATALOG"] = dict.fromkeys(ifos, catalog)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(
            {
                "key": key,
                "segments": {
                    cat: {
                        ifo: np.asarray(segs).tolist()
                        for ifo, segs in ifo_segs.items()
                    }
                    for cat, ifo_segs in segments.items()
                },
            },
            f,
        )
    return segments


def compute_veto_masks(
    events: "EventSet",
    categories: Sequence[VETO_CATEGORIES],
    ifos: Sequence[str],
    segments: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Compute a per-category veto mask against `events`.

    Args:
        events: the ledger to mask, e.g. a background `EventSet` or
            foreground `RecoveredInjectionSet`.
        categories: veto categories to compute masks for.
        ifos: interferometers to check; a category vetoes an event if
            it falls in a vetoed segment for *any* of them.
        segments: `{category: {ifo: (N, 2) array}}`, e.g. from
            `load_or_fetch_segments`.
    """
    masks = {}
    for cat in categories:
        mask = np.zeros(len(events), dtype=bool)
        for i, ifo in enumerate(ifos):
            ifo_segments = segments[cat][ifo]
            if len(ifo_segments):
                times = events.detection_time + events.shift[:, i]
                mask |= veto_mask(times, ifo_segments)
        logging.info(f"\t{mask.sum()} events removed for category {cat}")
        masks[cat] = mask
    return masks
