#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logging module for FixMS"""

import io
import logging


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
) -> logging.Logger:
    """Will construct a logger object.

    Args:
        name (str, optional): Name of the logger to attempt to use. This is ignored if in a prefect flowrun. Defaults to 'arrakis'.
        attach_handler (bool, optional): Attacjes a custom StreamHandler. Defaults to True.

    Returns:
        logging.Logger: The appropriate logger
    """
    logging.captureWarnings(True)
    logger = logging.getLogger(name)
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
