import importlib
from pathlib import Path
from typing import Union

import luigi
from cloudpathlib import CloudPath

PATH_LIKE = Union[CloudPath, Path, str]


class PathParameter(luigi.Parameter):
    """
    luigi `Parameter` class that handles parsing strings
    into pathlib.Path (or cloudpathlib.S3Path) objects
    """

    def parse(self, x: PATH_LIKE):
        if isinstance(x, (Path, CloudPath)):
            return x / ""
        if isinstance(x, str):
            if x.startswith("s3://"):
                x = CloudPath(x) / ""
            else:
                x = Path(x) / ""
        else:
            raise ValueError(
                f"Expected string, Path, or CloudPath, got {type(x)}"
            )

    def serialize(self, x):
        return str(x)

    def normalize(self, x):
        return self.parse(x)


def load_prior(path: str):
    """
    Imports a python path
    """
    module_path, prior = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    prior = getattr(module, prior)
    return prior
