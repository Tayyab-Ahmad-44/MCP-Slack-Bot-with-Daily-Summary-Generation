
import os
import logging

from core.config import settings


def configure_logging():
    """Logging File Configuration"""

    os.makedirs(settings.LOGGING_DIR, exist_ok=True)

    logging.basicConfig(
        filename="./logs/app.log",
        level=logging.INFO,

        # Format log messages
        format="%(asctime)s - %(levelname)s - %(message)s"
    )