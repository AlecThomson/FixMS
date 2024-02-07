#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logging module for FixMS"""

import io
import logging
import sys
from importlib.metadata import version
from typing import List

import astropy.units as u
from astropy.time import Time
from casacore.tables import table


def update_history(
    ms: str,
    message: str,
    app_params: List[str] = [],
    obs_id: int = 0,
    priority: str = "NORMAL",
) -> None:
    _allowed_priorities = ("DEBUGGING", "WARN", "NORMAL", "SEVERE")
    if priority not in _allowed_priorities:
        raise ValueError(
            f"Priority must be one of {_allowed_priorities}, got {priority}"
        )

    this_program = f"fixms-{version('fixms')}"
    now = (Time.now().mjd * u.day).to(u.second).value
    cli_args = sys.argv
    history_row = {
        "TIME": now,
        "OBSERVATION_ID": obs_id,
        "MESSAGE": message,
        "PRIORITY": priority,
        "CLI_COMMAND": cli_args,
        "APP_PARAMS": app_params,
        "ORIGIN": this_program,
    }
    with table(f"{ms}/HISTORY", readonly=False) as history:
        history.addrows(1)
        for key, value in history_row.items():
            history.putcell(key, history.nrows() - 1, value)


class LoggerWithHistory(logging.Logger):
    """Custom logger that will also update the HISTORY table in the MS."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def info(self, message, *args, ms=None, app_params=[], **kwargs):
        super().info(message, *args, **kwargs)
        if ms is not None:
            update_history(ms, message, app_params, priority="NORMAL")

    def warning(self, message, *args, ms=None, app_params=[], **kwargs):
        super().warning(message, *args, **kwargs)
        if ms is not None:
            update_history(ms, message, app_params, priority="WARN")

    def error(self, message, *args, ms=None, app_params=[], **kwargs):
        super().error(message, *args, **kwargs)
        if ms is not None:
            update_history(ms, message, app_params, priority="SEVERE")

    def critical(self, message, *args, ms=None, app_params=[], **kwargs):
        super().error(message, *args, **kwargs)
        if ms is not None:
            update_history(ms, message, app_params, priority="SEVERE")

    def debug(self, message, *args, ms=None, app_params=[], **kwargs):
        super().debug(message, *args, **kwargs)
        if ms is not None:
            update_history(ms, message, app_params, priority="DEBUGGING")

    def log(self, level, message, *args, ms=None, app_params=[], **kwargs):
        super().log(level, message, *args, **kwargs)
        if ms is not None:
            update_history(ms, message, app_params, priority=level)


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


# Create formatter
# formatter = logging.Formatter(
#     "SPICE: %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"
# )
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s.%(msecs)03d %(module)s - %(funcName)s: %(message)s"

    FORMATS = {
        logging.DEBUG: f"{blue}FixMS-%(levelname)s{reset} {format_str}",
        logging.INFO: f"{green}FixMS-%(levelname)s{reset} {format_str}",
        logging.WARNING: f"{yellow}FixMS-%(levelname)s{reset} {format_str}",
        logging.ERROR: f"{red}FixMS-%(levelname)s{reset} {format_str}",
        logging.CRITICAL: f"{bold_red}FixMS-%(levelname)s{reset} {format_str}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_fixms_logger(
    name: str = "fixms", attach_handler: bool = True
) -> LoggerWithHistory:
    """Will construct a logger object.

    Args:
        name (str, optional): Name of the logger to attempt to use. This is ignored if in a prefect flowrun. Defaults to 'arrakis'.
        attach_handler (bool, optional): Attacjes a custom StreamHandler. Defaults to True.

    Returns:
        logging.Logger: The appropriate logger
    """
    logging.captureWarnings(True)
    logger = LoggerWithHistory(name)
    logger.setLevel(logging.WARNING)

    if attach_handler:
        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Add formatter to ch
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

    return logger


logger = get_fixms_logger()
