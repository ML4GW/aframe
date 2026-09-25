import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from jsonargparse import ArgumentParser
from spython.main import Client

from scripts.env_hash import env_hash, local_libs

# Define the directory where the projects are located
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
BASE_DIR: Path = ROOT_DIR / "projects"
LIBS_DIR: Path = ROOT_DIR / "libs"
TEMPLATES_DIR: Path = ROOT_DIR / "container_templates"

# List of all available project names
PROJECTS: list[str] = [x.name for x in BASE_DIR.iterdir() if x.is_dir()]

# Extras to install into each project's container.
# Currently only needed for `data`, which uses extras to keep CUDA-torch
# out of its container.
EXTRAS: dict[str, list[str]] = {"data": ["cpu"]}

# Clear out the tools used to build the environment once complete to shrink
# container size. A project that needs a compiler at run time should install
# it from its apptainer.post, which marks it manually installed and so
# exempts it from the autoremove. See projects/infer for an example.
PURGE_BUILD_TOOLS = """# the toolchain was only needed to build the \
dependencies above
apt-get purge -y build-essential
apt-get autoremove -y
rm -rf /var/lib/apt/lists/*"""

COPY_EXCLUDE: set[str] = {
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "apptainer.def",
}


def _copy_entries(
    source_prefix: str, directory: Path, destination: str
) -> list[str]:
    """
    Build %files lines that copy a directory's contents entry by entry.
    """
    names = sorted(
        path.name
        for path in directory.iterdir()
        if path.name not in COPY_EXCLUDE
    )
    return [f"{source_prefix}{name} {destination}/{name}" for name in names]


def _get_files_block(project_name: str) -> str:
    """
    Build the %files block for an apptainer definition: the project itself,
    the `libs/` packages it depends on, and the root pyproject and lockfile
    that uv resolves the workspace against.
    """
    lines = _copy_entries(
        "",
        BASE_DIR / project_name,
        f"/opt/aframe/projects/{project_name}",
    )
    for lib in local_libs(project_name):
        lines.extend(
            _copy_entries(
                f"../../libs/{lib}/",
                LIBS_DIR / lib,
                f"/opt/aframe/libs/{lib}",
            )
        )

    lines.append("../../pyproject.toml /opt/aframe/pyproject.toml")
    lines.append("../../uv.lock /opt/aframe/uv.lock")

    return "\n".join(lines)


def _get_uv_command(project_name: str, subcommand: str) -> str:
    """
    Build a `uv sync`/`uv export` command for a project. The `test`
    group is installed so that CI can run tests inside the container.
    """
    cmd = (
        f"uv {subcommand} --frozen --no-default-groups --group test"
        f" --package {project_name}"
    )
    if subcommand == "export":
        # Need to use the pylock format here rather than requirements.txt
        # so that the index each package was locked from gets recorded.
        cmd += " --format pylock.toml"
    for extra in EXTRAS.get(project_name, []):
        cmd += f" --extra {extra}"
    return cmd


def create_definition_file(project_name: str) -> Path:
    """
    Create the apptainer definition file for a project from the appropriate
    template and write it to projects/<project>/apptainer.def.

    Projects with a <project>.conda-lock.yml use the micromamba template; all
    others use the uv template.
    """
    project_dir = BASE_DIR / project_name
    is_micromamba = (project_dir / f"{project_name}.conda-lock.yml").exists()
    template_name = "micromamba.def" if is_micromamba else "uv.def"
    template_text = (TEMPLATES_DIR / template_name).read_text()

    files_block = _get_files_block(project_name)
    # Optional per-project hooks
    post_file = project_dir / "apptainer.post"
    extra_post = post_file.read_text() if post_file.exists() else ""
    env_file = project_dir / "apptainer.env"
    extra_env = env_file.read_text() if env_file.exists() else ""

    # the micromamba image installs into an existing conda env, so it exports
    # a requirements file and pip-installs it. The uv image syncs directly.
    subcommand = "export" if is_micromamba else "sync"

    definition_text = (
        template_text.replace("@@PROJECT@@", project_name)
        .replace("@@FILES_BLOCK@@", files_block)
        .replace("@@UV_CMD@@", _get_uv_command(project_name, subcommand))
        .replace("@@EXTRA_POST@@", extra_post)
        .replace("@@EXTRA_ENV@@", extra_env)
        .replace("@@PURGE_BUILD_TOOLS@@", PURGE_BUILD_TOOLS)
        .replace("@@ENV_HASH@@", env_hash(project_name))
    )

    output_path = project_dir / "apptainer.def"
    output_path.write_text(definition_text)
    logging.info(f"Wrote template {template_name} to {output_path}")
    return output_path


def build_container(project_name: str, container_root: Path) -> str:
    project_path = BASE_DIR / project_name
    container_path = container_root / f"{project_name}.sif"

    create_definition_file(project_name)
    definition_path = project_path / "apptainer.def"

    # change directory to project path since
    # that's the root from where
    # the apptainer def files are defined
    cwd = os.getcwd()
    os.chdir(project_path)

    # build the container
    image, cmd = Client.build(
        image=str(container_path),
        recipe=str(definition_path),
        sudo=False,
        options=["--force"],
        stream=True,
    )
    try:
        for line in cmd:
            logging.info(line)
        return f"Successfully built container for {project_name}"
    except Exception as e:
        return f"Failed to build container for {project_name}: {e}"
    finally:
        os.chdir(cwd)


def validate_projects(projects: list[str]) -> None:
    invalid = [p for p in projects if p not in PROJECTS]
    if invalid:
        raise ValueError(
            f"Specified invalid projects: {', '.join(invalid)}. "
            f"The available projects are: {', '.join(PROJECTS)}"
        )


def build(projects: list[str], container_root: Path, max_workers: int) -> None:
    if not container_root:
        logging.info("Container root path is not set.")
        return

    failed_projects = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(build_container, project, container_root): project
            for project in projects
        }
        for future in as_completed(futures):
            project = futures[future]
            result = future.result()
            logging.info(result)
            if result.startswith("Failed"):
                failed_projects.append(project)

    logging.info("Container build summary:")
    if len(failed_projects) > 0:
        logging.error(
            f"Failed to build containers for the following projects: "
            f"{', '.join(failed_projects)}\n"
            f"To retry building these containers, run the following: "
            f"uv run build-containers {' '.join(failed_projects)}"
        )
    else:
        logging.info("All containers built successfully")


def main():
    parser = ArgumentParser(
        description="Automatically rebuild Aframe "
        "apptainer images for sub-projects"
    )

    parser.add_argument(
        "projects",
        nargs="*",
        default=PROJECTS,
        help="List of projects to build. "
        f"Default is all: {', '.join(PROJECTS)}",
    )

    parser.add_argument(
        "--container-root",
        type=Path,
        default=Path(os.getenv("AFRAME_CONTAINER_ROOT", "")),
        help="Path to the container root directory. "
        "Defaults to the $AFRAME_CONTAINER_ROOT environment variable.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent builds. Can be useful to set if "
        "your local TMPDIR is being overfilled when building containers. "
        "Default is `None`.",
    )

    parser.add_argument(
        "--definition-only",
        action="store_true",
        default=False,
        help="Write the definition files(s) for the specified project, and "
        "do not build the container images.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    validate_projects(args.projects)

    if args.definition_only:
        for project in args.projects:
            create_definition_file(project)
        return

    build(args.projects, args.container_root, args.max_workers)


if __name__ == "__main__":
    main()
