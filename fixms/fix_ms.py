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

logger = logging.getLogger(__name__)
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
        "--data-column", type=str, default="DATA", help="The column to fix"
    )
    parser.add_argument(
        "--corrected-data-column",
        type=str,
        default="CORRECTED_DATA",
        help="The column to write the corrected data to",
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
    )


if __name__ == "__main__":
    cli()
