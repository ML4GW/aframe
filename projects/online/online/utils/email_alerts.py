import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from lal import gpstime

SENDER_EMAIL = "aframe-online"


def send_email(message: MIMEMultipart, recipients: list[str]):
    try:
        # Connect to server and send email
        server = smtplib.SMTP()
        server.connect()
        server.send_message(message)
        server.quit()
        logging.info(f"Error notification sent to {', '.join(recipients)}")
    except Exception as e:
        logging.error(f"Failed to send error notification email: {str(e)}")


def send_error_email(
    name: str, error: str, traceback: str, recipients: list[str]
):
    now = gpstime.gps_time_now()
    date = gpstime.gps_to_utc(now)

    # Create message
    message = MIMEMultipart()
    message["Subject"] = f"Error Alert: {name} Process Failed"
    message["From"] = SENDER_EMAIL
    message["To"] = ", ".join(recipients)

    # Email body
    body = f"""
    An error occurred in subprocess: {name} at {date} UTC

    Error message: {str(error)}

    Traceback:
    {traceback}

    Aframe online deployment will be automatically relaunched.
    """

    message.attach(MIMEText(body, "plain"))

    send_email(message, recipients)


def send_init_email(recipients: list[str], outdir):
    message = MIMEMultipart()
    message["Subject"] = "Aframe Online Initialized"
    message["From"] = SENDER_EMAIL
    message["To"] = ", ".join(recipients)
    now = gpstime.gps_time_now()
    date = gpstime.gps_to_utc(now)

    # Email body
    body = f"""
    Aframe online deployment has been initialized at {date} UTC

    The run directory is {outdir}
    """

    message.attach(MIMEText(body, "plain"))

    send_email(message, recipients)
