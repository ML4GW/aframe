import time
import os
import logging
from .utils import subprocess_wrapper, run_subprocess_with_logging

logger = logging.getLogger("authenticate-subprocess")


def authenticate(minsecs: float = 1000, debug: bool = False):
    args = [
        "kinit",
        "aframe-1-scitoken/robot/ldas-pcdev12.ligo.caltech.edu@LIGO.ORG",
        "-k",
        "-t",
        os.path.expanduser(
            "~/robot/aframe-1-scitoken_robot_ldas-pcdev12.ligo.caltech.edu.keytab"  # noqa
        ),
    ]
    run_subprocess_with_logging(
        args, logger=logger, log_stderr_on_success=False
    )

    args = [
        "htgettoken",
        "-v",
        "-a",
        "vault.ligo.org",
        "-i",
        "igwn",
        "-r",
        "aframe-1-scitoken",
        "--scopes=gracedb.read",
        "--credkey=aframe-1-scitoken/robot/ldas-pcdev12.ligo.caltech.edu",
        f"--minsecs={minsecs}",
        "--nooidc",
    ]

    if debug:
        args.append("-d")

    run_subprocess_with_logging(
        args, logger=logger, log_stderr_on_success=False
    )


@subprocess_wrapper
def authenticate_subprocess(
    refresh: int, minsecs: float = 1000, debug: bool = False
):
    """
    Authentication subprocess loop that will re-authenticate
    every `refresh` seconds

    """
    logger.info("authenticate subprocess initialized")
    last_auth = time.time()
    while True:
        time.sleep(1e-1)
        if time.time() - last_auth > refresh:
            logger.info("Authenticating...")
            authenticate(minsecs, debug)
            last_auth = time.time()
            logger.info("Authentication complete")
