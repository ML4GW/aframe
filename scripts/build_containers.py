import logging
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from jsonargparse import ArgumentParser
from spython.main import Client

# Define the directory where the projects are located
BASE_DIR: Path = Path(__file__).resolve().parent.parent / "projects"
TEMPLATES_DIR: Path = (
    Path(__file__).resolve().parent.parent / "container_templates"
)

# List of all available project names
PROJECTS: list[str] = [x.name for x in BASE_DIR.iterdir() if x.is_dir()]


def _get_files_block(project_name: str) -> str:
    """
    Build the %files block for an apptainer definition by reading the
    project's [tool.uv.sources] and resolving which local paths to copy.
    """
    pyproject_path = BASE_DIR / project_name / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    sources = data["tool"]["uv"]["sources"]

    lines = [f". /opt/aframe/projects/{project_name}/"]
    seen: set[str] = set()

    for source in sources.values():
        path = source.get("path", "")
        if not path:
            # git source
            continue

        normalized = path.rstrip("/")

        lib_name = normalized.removeprefix("../../libs/")
        entry = f"../../libs/{lib_name} /opt/aframe/libs/{lib_name}"

        if entry not in seen:
            seen.add(entry)
            lines.append(entry)

    lines.append("../../pyproject.toml /opt/aframe/pyproject.toml")
    return "\n".join(lines)


def create_definition_file(project_name: str) -> Path:
    """
    Create the apptainer definition file for a project from the appropriate
    template and write it to projects/<project>/apptainer.def.

    Projects with a conda-lock.yml use the micromamba template; all others
    use the uv template.
    """
    project_dir = BASE_DIR / project_name
    is_micromamba = (project_dir / "conda-lock.yml").exists()
    template_name = "micromamba.def" if is_micromamba else "uv.def"
    template_text = (TEMPLATES_DIR / template_name).read_text()

    files_block = _get_files_block(project_name)
    definition_text = template_text.replace(
        "@@PROJECT@@", project_name
    ).replace("@@FILES_BLOCK@@", files_block)

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
