import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from jsonargparse import ArgumentParser
from spython.main import Client

# Define the directory where the projects are located
BASE_DIR: Path = Path(__file__).resolve().parent.parent / "projects"

# List of all available project names
PROJECTS: List[str] = [x.name for x in BASE_DIR.iterdir() if x.is_dir()]


def build_container(project_name: str, container_root: Path) -> str:
    project_path = BASE_DIR / project_name
    container_path = container_root / f"{project_name}.sif"
    definition_path = project_path / "apptainer.def"
    # change directory to project path since
    # that's the root from where
    # the apptainer def files are defined
    cwd = os.getcwd()
    os.chdir(project_path)

    # skip this project if there
    # is no associated apptainer definition file
    if not definition_path.exists():
        out = (
            f"Apptainer definition file for {project_name} "
            "does not exist. Skipping build."
        )
        return out

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


def validate_projects(projects: List[str]) -> None:
    invalid = [p for p in projects if p not in PROJECTS]
    if invalid:
        raise ValueError(
            f"Specified invalid projects: {', '.join(invalid)}. "
            f"The available projects are: {', '.join(PROJECTS)}"
        )


def build(projects: List[str], container_root: Path, max_workers: int) -> None:
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
            f"poetry run build-containers {' '.join(failed_projects)}"
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
        default=None,  # Set a default number of workers
        help="Maximum number of concurrent builds. Can be useful to set if "
        "your local TMPDIR is being overfilled when building containers. "
        "Default is `None`.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    validate_projects(args.projects)
    build(args.projects, args.container_root, args.max_workers)


if __name__ == "__main__":
    main()
