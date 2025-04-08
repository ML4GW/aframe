import time
import os
import logging
from .utils import subprocess_wrapper, run_subprocess_with_logging

logger = logging.getLogger("authenticate-subprocess")


def authenticate(debug: bool = False):
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
        "--nooidc",
    ]

    if debug:
        args.append("-d")

    run_subprocess_with_logging(
        args, logger=logger, log_stderr_on_success=False
    )


@subprocess_wrapper
def authenticate_subprocess(refresh: int, debug: bool = False):
    """
    Authentication subprocess loop that will re-authenticate
    every `refresh` seconds

    """
    logger.info("authenticate subprocess initialized")
    last_auth = time.time() - refresh - 5
    while True:
        time.sleep(1e-1)
        if time.time() - last_auth > refresh:
            logger.info("Authenticating...")
            authenticate(debug)
            last_auth = time.time()
            logger.info("Authentication complete")
