"""Hash the inputs that determine a project's container environment.

Project and library sources are installed in editable mode, so an
image only has to be updated when the environment changes. Images
record this hash at build time and the pipeline compares it with
the local repo's to warn about stale images.
"""

import hashlib
import sys
import tomllib
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
PROJECTS_DIR: Path = ROOT_DIR / "projects"
LIBS_DIR: Path = ROOT_DIR / "libs"
TEMPLATES_DIR: Path = ROOT_DIR / "container_templates"


def local_libs(project: str) -> list[str]:
    """
    The `libs/` that a project depends on, including those that enter via
    a library's own dependencies.
    """
    seen: set[str] = set()
    pyproject_files: list[Path] = [PROJECTS_DIR / project / "pyproject.toml"]

    while len(pyproject_files) > 0:
        with open(pyproject_files.pop(), "rb") as f:
            data = tomllib.load(f)

        sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        for name, source in sources.items():
            # index pins like torch are lists, not dicts.
            # git sources are dicts but don't have a `workspace` key
            if not isinstance(source, dict) or not source.get("workspace"):
                continue
            if name in seen:
                continue
            seen.add(name)
            pyproject_files.append(LIBS_DIR / name / "pyproject.toml")

    return sorted(seen)


def env_files(project: str) -> list[Path]:
    """Every file whose contents impact the project's environment."""
    project_dir = PROJECTS_DIR / project
    conda_lock = project_dir / f"{project}.conda-lock.yml"
    template = "micromamba.def" if conda_lock.exists() else "uv.def"
    candidates = [
        ROOT_DIR / "uv.lock",
        ROOT_DIR / "pyproject.toml",
        ROOT_DIR / "scripts" / "build_containers.py",
        TEMPLATES_DIR / template,
        project_dir / "pyproject.toml",
        project_dir / "apptainer.post",
        project_dir / "apptainer.env",
        conda_lock,
        *(LIBS_DIR / lib / "pyproject.toml" for lib in local_libs(project)),
    ]
    return [path for path in candidates if path.exists()]


def manifest(project: str) -> str:
    """`sha256sum`-style lines for each environment file, sorted by path."""
    paths = sorted(str(p.relative_to(ROOT_DIR)) for p in env_files(project))
    return "".join(
        f"{hashlib.sha256((ROOT_DIR / p).read_bytes()).hexdigest()}  {p}\n"
        for p in paths
    )


def env_hash(project: str) -> str:
    """A short hash of the files that impact the project's environment.

    Hashes the manifest, so from the repo root this equals the first 12
    characters of `sha256sum <files in order> | sha256sum`.
    """
    return hashlib.sha256(manifest(project).encode()).hexdigest()[:12]


if __name__ == "__main__":
    print(env_hash(sys.argv[1]))  # noqa: T201
