import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging


def send_error_email(
    name: str, error: str, traceback: str, recipients: list[str]
):
    sender_email = "aframe-online"

    # Create message
    message = MIMEMultipart()
    message["Subject"] = f"Error Alert: {name} Process Failed"
    message["From"] = sender_email
    message["To"] = ", ".join(recipients)

    # Email body
    body = f"""
    An error occurred in subprocess: {name}

    Error message: {str(error)}

    Traceback:
    {traceback}

    Aframe online deployment will be automatically relaunched.
    """

    message.attach(MIMEText(body, "plain"))

    try:
        # Connect to server and send email
        server = smtplib.SMTP()
        server.connect()
        server.send_message(message)
        server.quit()
        logging.info(f"Error notification sent to {', '.join(recipients)}")
    except Exception as e:
        logging.error(f"Failed to send error notification email: {str(e)}")
