import os

import luigi


class ModelRepositoryTarget(luigi.Target):
    """
    Target for checking the existence of an aframe model repository
    """

    def __init__(self, path, version: int = -1):
        super().__init__()
        self.path = path
        self.version = version

    def get_versions(self):
        versions = []

        # currently only check `aframe` model for versions
        # since this is the most important
        version_dir = os.path.join(self.path, "aframe")
        if not os.path.isdir(version_dir):
            return versions

        # filter directories that can be converted to integers and find the max
        for path in os.listdir(version_dir):
            subdir = os.path.join(version_dir, path)
            if os.path.isdir(subdir) and path.isdigit():
                versions.append(path)
        return versions

    def exists(self):
        # now that we know the "aframe" directory
        # exists, parse the versions. If none exist,
        # return False
        versions = self.get_versions()
        if not versions:
            return False

        version = self.version
        if version == -1:
            version = max(versions)

        version = str(version)

        # define the structure of the model repo
        structure = {
            "aframe": ["config.pbtxt", (version, ["model.plan"])],
            "aframe-stream": ["config.pbtxt", (version, ["model.empty"])],
            "preprocessor": ["config.pbtxt", (version, ["model.pt"])],
            "snapshotter": ["config.pbtxt", (version, ["model.onnx"])],
        }

        # check each part of the repo structure exists
        for directory, contents in structure.items():
            dir_path = os.path.join(self.path, directory)
            if not os.path.isdir(dir_path):
                return False

            for item in contents:
                if isinstance(item, str):
                    # Check for files directly under the directory
                    if not os.path.isfile(os.path.join(dir_path, item)):
                        return False
                elif isinstance(item, tuple):
                    # Check for subdirectories and their files
                    sub_dir, sub_files = item
                    sub_dir_path = os.path.join(dir_path, sub_dir)
                    if not os.path.isdir(sub_dir_path):
                        return False
                    for sub_file in sub_files:
                        if not os.path.isfile(
                            os.path.join(sub_dir_path, sub_file)
                        ):
                            return False

        # If all checks passed
        return True

    def complete(self):
        return self.exists()
