import time
import subprocess
import os
import logging
from .utils import subprocess_wrapper

logger = logging.getLogger("authenticate-subprocess")


def authenticate():
    # TODO: don't hardcode keytab locations
    subprocess.run(
        [
            "kinit",
            "aframe-1-scitoken/robot/ldas-pcdev12.ligo.caltech.edu@LIGO.ORG",
            "-k",
            "-t",
            os.path.expanduser(
                "~/robot/aframe-1-scitoken_robot_ldas-pcdev12.ligo.caltech.edu.keytab"  # noqa
            ),
        ]
    )
    subprocess.run(
        [
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
    )


@subprocess_wrapper
def authenticate_subprocess(refresh: int = 1000):
    """
    Authentication subprocess loop that will re-authenticate
    every `refresh` seconds

    """
    logger.info("authenticate subprocess initialized")
    last_auth = time.time() + refresh + 5
    while True:
        if last_auth - time.time() > refresh:
            logger.debug("Authenticating...")
            authenticate()
            last_auth = time.time()
            logger.debug("Authentication complete")
