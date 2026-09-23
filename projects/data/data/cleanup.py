from collections.abc import Iterable
from pathlib import Path


def remove_empty_dirs(files: Iterable[Path], root: Path) -> None:
    """Remove the directories that held `files`, up to but excluding `root`.

    Aggregating with `clean=True` deletes each per-branch file once it is
    merged, which leaves its directory behind. Only empty directories are
    removed, so anything still holding files is left alone.
    """
    root = Path(root)
    dirs = {Path(f).parent for f in files}
    for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        while d != root and d.is_relative_to(root):
            try:
                d.rmdir()
            except OSError:
                # not empty, or already removed
                break
            d = d.parent
