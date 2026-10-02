"""Copy the local repo into code/ in the working directory for the run to use

A run and its condor jobs execute the copied code, so edits to the local repo
take effect when the run is next started, not while it runs. The copied code
is read-only to prevent any untracked modifications to the code. The copy
holds everything tracked by git.

Standard library only, so that it runs before any environment is set up.
"""

import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git_files(repo: Path, *options: str) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", *options], text=True
    )
    # The "-z" flag makes the delimiter the NUL byte
    return [Path(f) for f in output.split("\0") if f]


def repo_files(repo: Path = REPO) -> list[Path]:
    return [f for f in git_files(repo, "--cached") if (repo / f).is_file()]


def snapshot(dest: Path = Path("code"), repo: Path = REPO) -> None:
    # Snakemake holds these while a run is active, and we don't want
    # to copy over the code during that time.
    locks = Path(".snakemake/locks")
    if locks.is_dir() and any(locks.iterdir()):
        sys.exit(
            "This run directory is locked by a running snakemake, so its "
            f"{dest}/ can't be replaced. If no run is going, remove the stale "
            f"locks with `rm {locks}/*` and retry."
        )
    new = git_files(repo, "--others", "--exclude-standard")
    if new:
        print(  # noqa: T201
            f"warning: not copying {len(new)} file(s) that git doesn't know "
            "about yet; `git add` any the run needs:\n  "
            + "\n  ".join(str(f) for f in new[:20])
            + ("\n  ..." if len(new) > 20 else ""),
            file=sys.stderr,
        )
    shutil.rmtree(dest, ignore_errors=True)
    for f in repo_files(repo):
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / f, dest / f)
        # Remove the write-bits from each file's permissions
        mode = (dest / f).stat().st_mode
        (dest / f).chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


if __name__ == "__main__":
    snapshot()
