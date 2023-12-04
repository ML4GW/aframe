import os
import shutil
import subprocess

_kinit_errs = {
    "Key table file '{keytab_location}' not found": (
        "kinit command failed because key table file "
        "'{keytab_location}' not found. See instructions for "
        "setting up passwordless authentication at "
        "https://computing.docs.ligo.org/guide/auth/kerberos/#ligo"
    ),
    "Keytab contains no suitable keys for {user}@LIGO.ORG": (
        "kinit command failed because key table file "
        "was not configured for user {user}. Make sure "
        "that when following instructions at "
        "https://computing.docs.ligo.org/guide/auth/kerberos/#ligo "
        "that you replace 'albert.einstein' with '{user}'."
    ),
    "Password incorrect while getting initial credentials": (
        "kinit command failed because password saved to "
        "key table file is incorrect for user {user}. Re-generate "
        "with the correct password using the instructions at "
        "https://computing.docs.ligo.org/guide/auth/kerberos/#ligo"
    ),
}


def _validate_env(env_var: str):
    """
    Validate the existence of an environment variable
    Args:
        env_var : str
            The name of the environment variable
    Raises:
        ValueError If the environment variable is not set

    Returns The value of the environment variable
    """
    value = os.getenv(env_var)
    if value is None:
        raise ValueError(f"{env_var} environment variable not set")

    return value


def _check_kinit_errs(stderr: str, user: str, keytab_location: str) -> None:
    for key, msg in _kinit_errs.items():
        key = key.format(user=user, keytab_location=keytab_location)
        if stderr.startswith("kinit: " + key):
            msg = msg.format(user=user, keytab_location=keytab_location)
            raise OSError(msg)


def kinit():
    user = _validate_env("LIGO_USERNAME")
    keytab_location = _validate_env("KRB5_KTNAME")

    kinit_command = shutil.which("kinit")
    if kinit_command is None:
        raise ValueError("kinit command not found")

    args = [
        kinit_command,
        "-p",
        f"{user}@LIGO.ORG",
        "-k",
        "-t",
        keytab_location,
    ]

    response = subprocess.run(args, capture_output=True, text=True)
    if response.returncode:
        # first check to see if we recognize any of the issues
        _check_kinit_errs(response.stderr, user, keytab_location)

        # if not, raise a generic error with kinit's full stderr
        raise RuntimeError(
            "kinit command failed with return code {}: {}".format(
                response.returncode, response.stderr
            )
        )


def make_cert(cert_path: str) -> None:
    from ciecplib.ui import get_cert
    from ciecplib.x509 import write_cert

    kinit()
    cert, key = get_cert(kerberos=True)
    write_cert(cert_path, cert, key)


def authenticate():
    """
    Authenticate a user to access LIGO data sources

    This function will load a X509 certificate from the environment
    variable `X509_USER_PROXY`. If the credential doesn't exist,
    it will create a new one. If the credential exists and is valid,
    it will continue to use it. Otherwise, it will generate a new credential.

    If generating new credential, a kerberos keytab is required
    for passwordless authentication. It's location should be
    specified in the environment variable `KRB5_KTNAME`.
    This function assumes the user has already generated a kerberos keytab
    with principal user.name@LIGO.ORG. This function will read in username
    from the environment variable `LIGO_USERNAME`

    For instructions on generating a kerberos keytab,
    see https://computing.docs.ligo.org/guide/auth/kerberos/

    """

    from ciecplib.x509 import check_cert, load_cert

    cert_path = _validate_env("X509_USER_PROXY")
    if os.path.exists(cert_path):
        cert = load_cert(cert_path)
        try:
            check_cert(cert)
        except RuntimeError:
            make_cert(cert_path)
    else:
        make_cert(cert_path)

    return cert_path
