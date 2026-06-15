"""Re-lock conda environments when their source or lock file changes.

Called by pre-commit with the changed files as arguments.
Assumes the convention: <stem>.yaml -> <stem>.conda-lock.yml
The --check-input-hash flag makes this a no-op if nothing has changed.
"""

import subprocess
import sys
from pathlib import Path


def main():
    seen = set()
    for path in map(Path, sys.argv[1:]):
        stem = path.name.removesuffix(".conda-lock.yml").removesuffix(".yaml")
        key = (path.parent, stem)
        if key in seen:
            continue
        seen.add(key)
        subprocess.run(
            [
                "conda-lock",
                "lock",
                "--file",
                str(path.parent / f"{stem}.yaml"),
                "--platform",
                "linux-64",
                "--lockfile",
                str(path.parent / f"{stem}.conda-lock.yml"),
                "--check-input-hash",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
