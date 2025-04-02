import time
import subprocess
import os
from online.subprocesses.wrapper import subprocess_wrapper


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
    # Re-authenticate every 1000 seconds so that
    # the scitoken doesn't expire. Doing it in this
    # loop as it's the earliest point of submission
    while True:
        last_auth = time.time()
        if last_auth - time.time() > refresh:
            authenticate()
            last_auth = time.time()
