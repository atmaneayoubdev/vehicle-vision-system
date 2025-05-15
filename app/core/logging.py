from loguru import logger
import sys


def configure_logger():
    # Remove the default logger
    logger.remove()

    # Add a single console logger for all levels
    logger.add(
        sys.stdout,
        level="DEBUG",  # Log all levels from DEBUG and above
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}: {message}",
        colorize=True
    )

    # Add a console logger for CRITICAL messages to stderr
    logger.add(
        sys.stderr,
        level="CRITICAL",  # Log CRITICAL messages
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}: {message}",
        colorize=True
    )
