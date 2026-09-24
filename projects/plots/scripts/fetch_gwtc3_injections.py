# ruff: noqa: F821
"""Copy the GWTC-3 sensitivity injection set into the run directory.

Executed via the snakemake `script:` directive, as a localrule, so the
download happens once on the submit node and is cached there rather
than in every job that needs it.
"""

import shutil
import sys

from astropy.utils.data import download_file
from plots.core.gwtc3 import INJECTION_URL

sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

# downloads to ~/.aframe/cache/ if not already there
fname = download_file(INJECTION_URL, cache=True, pkgname="aframe")
shutil.copyfile(fname, snakemake.output[0])
