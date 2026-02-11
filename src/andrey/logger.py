"""Colored terminal logging setup with optional file logging."""

import logging
import sys
from pathlib import Path
from typing import Optional


class ColorFormatter(logging.Formatter):
    """Terminal-friendly colored log formatter."""

    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname:<7}{self.RESET}"
        return super().format(record)


def setup_logging(
    verbose: bool = False, log_dir: Optional[str] = None
) -> None:
    """Configure structured logging to stderr and optionally to a file.

    Args:
        verbose: If True, set log level to DEBUG for stderr.
        log_dir: If provided, write a session.log file in this directory
            at DEBUG level (captures all messages regardless of verbose flag).
    """
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger("andrey")
    root.setLevel(logging.DEBUG)  # allow all levels, handlers filter

    # Stderr handler (colored)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(
        ColorFormatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(stderr_handler)

    # File handler (always DEBUG, plain text)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            str(log_path / "session.log"), mode="w"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(file_handler)

    # Suppress noisy library loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
