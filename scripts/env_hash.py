"""Hash the inputs that determine a project's container environment.

Project and library sources are installed in editable mode, so an
image only has to be updated when the environment changes. Images
record this hash at build time and the pipeline compares it with
the local repo's to warn about stale images.

Only the packages the project's image installed are hashed so that
a lock change in one project doesn't impact other projects.

Standard library only so that the Snakefile can import it.
"""

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
PROJECTS_DIR: Path = ROOT_DIR / "projects"
LIBS_DIR: Path = ROOT_DIR / "libs"
TEMPLATES_DIR: Path = ROOT_DIR / "container_templates"
LOCK_FILE: Path = ROOT_DIR / "uv.lock"

# Extras to install into each project's container.
# Currently only needed for `data`, which uses extras to keep CUDA-torch
# out of its container.
EXTRAS: dict[str, list[str]] = {"data": ["cpu"]}

# Dependency groups installed into every container, so that CI can run
# tests inside it
GROUPS: list[str] = ["test"]


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


def uv_command(project: str, subcommand: str) -> list[str]:
    """
    The `uv sync`/`uv export` command that installs a project's environment.
    """
    args = ["uv", subcommand, "--frozen", "--no-default-groups"]
    for group in GROUPS:
        args += ["--group", group]
    args += ["--package", project]
    for extra in EXTRAS.get(project, []):
        args += ["--extra", extra]
    return args


def locked_requirements(project: str) -> bytes:
    """The packages the project's image installs, pinned with their hashes,
    as `uv export` resolves them from `uv.lock`.
    """
    args = [*uv_command(project, "export"), "--no-header", "--no-annotate"]
    return subprocess.check_output(args, cwd=ROOT_DIR)


def uv_settings() -> str:
    """The root `pyproject.toml`'s `[tool.uv]` table, which can change an
    install without changing the lock (e.g. build settings). The rest of the
    file doesn't affect images.
    """
    with open(ROOT_DIR / "pyproject.toml", "rb") as f:
        settings = tomllib.load(f)["tool"]["uv"]
    return json.dumps(settings, sort_keys=True)


def env_files(project: str) -> list[Path]:
    """Every file, besides `uv.lock` and the root `pyproject.toml`, whose
    contents impact the project's environment.
    """
    project_dir = PROJECTS_DIR / project
    conda_lock = project_dir / f"{project}.conda-lock.yml"
    template = "micromamba.def" if conda_lock.exists() else "uv.def"
    candidates = [
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
    """`sha256sum`-style lines for each environment file, sorted by path,
    then the project's packages from `uv.lock` and the root `[tool.uv]`.
    """

    def line(content: bytes, name: str) -> str:
        return f"{hashlib.sha256(content).hexdigest()}  {name}\n"

    paths = sorted(str(p.relative_to(ROOT_DIR)) for p in env_files(project))
    lines = [line((ROOT_DIR / p).read_bytes(), p) for p in paths]
    lines.append(line(locked_requirements(project), "uv.lock"))
    lines.append(line(uv_settings().encode(), "pyproject.toml [tool.uv]"))
    return "".join(lines)


def env_hash(project: str) -> str:
    """A short hash of the project's environment: the first 12 characters
    of the sha256 of its manifest.
    """
    return hashlib.sha256(manifest(project).encode()).hexdigest()[:12]


def build_commit(project: str) -> str:
    """The local repo's commit, with `-dirty` if any of the project's
    environment files, `uv.lock` or the root `pyproject.toml` differ from it.
    """
    root_files = [LOCK_FILE, ROOT_DIR / "pyproject.toml"]
    paths = [str(p) for p in [*env_files(project), *root_files]]

    def git(*args: str) -> str:
        cmd = ["git", *args]
        return subprocess.check_output(cmd, cwd=ROOT_DIR, text=True).strip()

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", *paths)
    return f"{commit}-dirty" if dirty else commit


if __name__ == "__main__":
    print(env_hash(sys.argv[1]))  # noqa: T201
