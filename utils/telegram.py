import requests
from dotenv import dotenv_values
from loguru import logger

config = dotenv_values(".env")


def send_message(message: str) -> dict:
    """Send a message to a telegram chat using the Telegram API.

    Args:
        message (str): The message to send

    Returns:
        dict: The response from the API
    """
    # Check if required environment variables exist
    if "TOKEN" not in config or "CHAT_ID" not in config:
        logger.warning(
            "Required environment variables 'TOKEN' or 'CHAT_ID' not found in .env file. Notification not sent."
        )
        return {}

    # Send the message using the Telegram API
    url = f"https://api.telegram.org/bot{config['TOKEN']}/sendMessage?chat_id={config['CHAT_ID']}&text={message}"
    return requests.get(url).json()
