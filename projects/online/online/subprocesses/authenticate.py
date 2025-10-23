import time
import os
import logging
from .utils import subprocess_wrapper, run_subprocess_with_logging

logger = logging.getLogger("authenticate-subprocess")

AFRAME_KEYTAB = os.getenv(
    "AFRAME_KEYTAB",
    "/home/aframe.online/robot/aframe-online_robot_aframe.ldas.cit.keytab",
)

AFRAME_CREDKEY = os.getenv(
    "AFRAME_CREDKEY", "aframe-online/robot/aframe.ldas.cit"
)


def authenticate(minsecs: float = 1000, debug: bool = False):
    args = [
        "kinit",
        AFRAME_CREDKEY + "@LIGO.ORG",
        "-k",
        "-t",
        AFRAME_KEYTAB,
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
        AFRAME_CREDKEY.split("/")[0],
        "--scopes=gracedb.read",
        f"--credkey={AFRAME_CREDKEY}",
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
