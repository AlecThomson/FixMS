#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to correct the ASKAP beam positions and apply a rotation
to apply a change of the reference frame of the visibilities
"""

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from fixms.fix_ms_corrs import fix_ms_corrs
from fixms.fix_ms_dir import fix_ms_dir
from fixms.logger import logger

logger.setLevel(logging.INFO)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "ms", help="Measurement set to update", type=str, default=None, nargs="?"
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000,
        help="The chunksize to use when reading the MS",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=1000,
        help="The maximum number of chunks to process at once",
    )
    parser.add_argument(
        "--data-column", type=str, default="DATA", help="The column to fix"
    )
    parser.add_argument(
        "--corrected-data-column",
        type=str,
        default="CORRECTED_DATA",
        help="The column to write the corrected data to",
    )
    parser.add_argument(
        "--no-fix-stokes-factor",
        dest="no_fix_stokes_factor",
        action="store_true",
        help="Don't fix the Stokes factor. Use this if you have *not* used ASKAPsoft. If you have used ASKAPsoft, you should leave this option alone.",
    )
    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    fix_ms_dir(args.ms)

    fix_ms_corrs(
        Path(args.ms),
        chunksize=args.chunksize,
        data_column=args.data_column,
        corrected_data_column=args.corrected_data_column,
        fix_stokes_factor=not args.no_fix_stokes_factor,
    )


if __name__ == "__main__":
    cli()
