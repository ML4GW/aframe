"""Publish container images to your OSDF staging directory, where condor
jobs fetch them and `container_source: osdf` finds them.

    python -m scripts.publish_images [data infer plots]

Run on a CIT AP with /osdf mounted. Copies each
`$AFRAME_CONTAINER_ROOT/<project>.sif` to `aframe-<project>-<hash>.sif`
in `/igwn/cit/staging/<USER>`, where `<hash>` is the environment hash
stamped into the image at build time. Only images built with their
environment files committed are published.

Jobs read images through OSDF caches, which keep serving the copy they
already hold even if the file at CIT is replaced. This means a name must
never be reused for a different image. Any environment change gets a new
hash, and an image already published under its hash is skipped.

Standard library only, so that the Snakefile can import it.
"""

import argparse
import getpass
import os
import shutil
import subprocess
from pathlib import Path

from scripts.env_hash import env_hash

# The images condor jobs can run in. train and export only run on the AP.
# Must match the CI size check in .github/workflows/project-build-test.yaml.
OSDF_PROJECTS = ("data", "infer", "plots")


def staging_dir(user: str | None = None) -> str:
    return f"/igwn/cit/staging/{user or getpass.getuser()}"


def image_name(project: str, env: str) -> str:
    return f"aframe-{project}-{env}.sif"


def stamp(image: Path, name: str) -> str | None:
    result = subprocess.run(
        ["apptainer", "exec", str(image), "cat", f"/opt/{name}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def publish(project: str, container_root: Path, dest_dir: Path) -> None:
    image = container_root / f"{project}.sif"
    if not image.exists():
        raise FileNotFoundError(
            f"{image} doesn't exist. Use `build_containers` to build it."
        )
    env, commit = stamp(image, "env_hash"), stamp(image, "build_commit")
    if env is None or commit is None:
        raise ValueError(
            f"{image} predates build stamps. Rebuild with `build_containers`."
        )
    if commit.endswith("-dirty"):
        raise ValueError(
            f"{image} was built with uncommitted changes to its environment "
            "files. Commit them, then rebuild it."
        )
    if env != env_hash(project):
        print(  # noqa: T201
            f"warning: {image} was built for environment {env}, not the "
            f"local repo's {env_hash(project)}."
        )

    dest = dest_dir / image_name(project, env)
    if dest.exists():
        print(f"{dest} already published")  # noqa: T201
        return
    print(f"publishing {image} (built from {commit}) to {dest}")  # noqa: T201
    try:
        shutil.copyfile(image, dest)
    except BaseException:
        # a partial copy would otherwise count as published next time
        dest.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish container images to OSDF staging"
    )
    parser.add_argument(
        "projects",
        nargs="*",
        choices=OSDF_PROJECTS,
        help=f"Default is all: {', '.join(OSDF_PROJECTS)}",
    )
    parser.add_argument(
        "--container_root",
        type=Path,
        default=os.getenv("AFRAME_CONTAINER_ROOT"),
    )
    args = parser.parse_args()
    if args.container_root is None:
        raise ValueError("set AFRAME_CONTAINER_ROOT or pass --container_root")
    dest_dir = Path("/osdf" + staging_dir())
    if not dest_dir.is_dir():
        raise FileNotFoundError(
            f"{dest_dir} doesn't exist. Use an AP with OSDF mounted."
        )
    for project in args.projects or OSDF_PROJECTS:
        publish(project, args.container_root, dest_dir)


if __name__ == "__main__":
    main()
