#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix the correlation rotation of ASKAP MSs.

Converts the ASKAP standard correlations to the 'standard' correlations
This will make them compatible with most imagers (e.g. wsclean, CASA)

The new correlations are placed in a new column called 'CORRECTED_DATA'
"""
__author__ = ["Alec Thomson"]
import logging
from pathlib import Path
from typing import Iterator, Optional

import astropy.units as u
import numpy as np
from casacore.tables import makecoldesc, table
from tqdm import tqdm

from fixms.logger import TqdmToLogger, logger

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)
logger.setLevel(logging.INFO)


def set_pol_axis(ms: Path, pol_ang: u.Quantity, feed_idx: Optional[int] = None) -> None:
    with table((ms / "FEED").as_posix(), readonly=True, ack=False) as tf:
        ms_feed = tf.getcol("RECEPTOR_ANGLE") * u.rad
        # PAF is at 45deg to feeds
        # 45 - feed_angle = pol_angle
        pol_axes = -(ms_feed - 45.0 * u.deg)

    # Backup the original RECEPTOR_ANGLE to INSTRUMENT_RECEPTOR_ANGLE
    with table((ms / "FEED").as_posix(), readonly=False, ack=False) as tf:
        tf.putcol("INSTRUMENT_RECEPTOR_ANGLE", ms_feed.to(u.rad).value)
        tf.flush()
    logger.info("Backed up the original RECEPTOR_ANGLE to INSTRUMENT_RECEPTOR_ANGLE")

    if feed_idx is None:
        assert (ms_feed[:, 0] == ms_feed[0, 0]).all() & (
            ms_feed[:, 1] == ms_feed[0, 1]
        ).all(), "The RECEPTOR_ANGLE changes with time, please check the MS"

        old_pol_ang = pol_axes[0, 0].to(u.deg)
    else:
        logger.debug(f"Extracting the third-axis orientation for {feed_idx=}")
        old_pol_ang = pol_axes[feed_idx, 0].to(u.deg)

    assert old_pol_ang == pol_ang, "Updated pol angle does not match the original!"

    pol_axes_cor = pol_axes - pol_ang
    # PAF is at 45deg to feeds
    # 45 - feed_angle = pol_angle
    # -> feed_angle = 45 - pol_angle
    new_ms_feed = -(pol_axes_cor - 45.0 * u.deg)
    with table((ms / "FEED").as_posix(), readonly=False, ack=False) as tf:
        tf.putcol("RECEPTOR_ANGLE", new_ms_feed.to(u.rad).value)
        tf.flush()


def get_pol_axis(ms: Path, feed_idx: Optional[int] = None) -> u.Quantity:
    """Get the polarization axis from the ASKAP MS. Checks are performed
    to ensure this polarisation axis angle is constant throughout the observation.


    Args:
        ms (Path): The path to the measurement set that will be inspected
        feed_idx (Optional[int], optional): Specify the entery in the FEED
        table of `ms` to return. This might be required when a subset of a
        measurement set has been extracted from an observation with a varying
        orientation.

    Returns:
        astropy.units.Quantity: The rotation of the PAF throughout the observing.
    """
    with table((ms / "FEED").as_posix(), readonly=True, ack=False) as tf:
        ms_feed = tf.getcol("RECEPTOR_ANGLE") * u.rad
        # PAF is at 45deg to feeds
        # 45 - feed_angle = pol_angle
        pol_axes = -(ms_feed - 45.0 * u.deg)

    if feed_idx is None:
        assert (ms_feed[:, 0] == ms_feed[0, 0]).all() & (
            ms_feed[:, 1] == ms_feed[0, 1]
        ).all(), "The RECEPTOR_ANGLE changes with time, please check the MS"

        pol_ang = pol_axes[0, 0].to(u.deg)
    else:
        logger.debug(f"Extracting the third-axis orientation for {feed_idx=}")
        pol_ang = pol_axes[feed_idx, 0].to(u.deg)

    return pol_ang


def convert_correlations(
    correlations: np.ndarray, pol_axis: u.Quantity, fix_stokes_factor: bool = True
) -> np.ndarray:
    """
    Convert ASKAP standard correlations to the 'standard' correlations

    Args:
        correlations (np.ndarray): The correlations from the MS. Has shape (NCOR, NCHAN, 4)
        pol_axis (astropy.units.Quantity): The polarization axis angle of the MS
        fix_stokes_factor (bool, optional): Whether to fix the Stokes factor. Defaults to True.

    Returns:
        np.ndarray: The correlations in the 'standard' format

    NOTES:
    In general, ASKAP forms Stokes I, Q, U, V as follows:

    .. code-block::

        ⎡I⎤   ⎡    1         0         0          1    ⎤ ⎡XXₐ⎤
        ⎢ ⎥   ⎢                                        ⎥ ⎢   ⎥
        ⎢Q⎥   ⎢sin(2⋅P)   cos(2⋅P)  cos(2⋅P)  -sin(2⋅P)⎥ ⎢XYₐ⎥
        ⎢ ⎥ = ⎢                                        ⎥⋅⎢   ⎥
        ⎢U⎥   ⎢-cos(2⋅P)  sin(2⋅P)  sin(2⋅P)  cos(2⋅P) ⎥ ⎢YXₐ⎥
        ⎢ ⎥   ⎢                                        ⎥ ⎢   ⎥
        ⎣V⎦   ⎣    0       -1.0⋅ⅈ    1.0⋅ⅈ        0    ⎦ ⎣YYₐ⎦


    Where P is the polarization axis angle. In the common case of P=-45deg this becomes:

    .. code-block::

        ⎡I⎤   ⎡1     0       0    1⎤ ⎡XXₐ⎤
        ⎢ ⎥   ⎢                    ⎥ ⎢   ⎥
        ⎢Q⎥   ⎢-1    0       0    1⎥ ⎢XYₐ⎥
        ⎢ ⎥ = ⎢                    ⎥⋅⎢   ⎥
        ⎢U⎥   ⎢0     -1     -1    0⎥ ⎢YXₐ⎥
        ⎢ ⎥   ⎢                    ⎥ ⎢   ⎥
        ⎣V⎦   ⎣0   -1.0⋅ⅈ  1.0⋅ⅈ  0⎦ ⎣YYₐ⎦

    or

    .. code-block::

        ⎡I⎤   ⎡    XXₐ + YYₐ     ⎤
        ⎢ ⎥   ⎢                  ⎥
        ⎢Q⎥   ⎢    -XXₐ + YYₐ    ⎥
        ⎢ ⎥ = ⎢                  ⎥
        ⎢U⎥   ⎢    -XYₐ - YXₐ    ⎥
        ⎢ ⎥   ⎢                  ⎥
        ⎣V⎦   ⎣-ⅈ⋅XYₐ + 1.0⋅ⅈ⋅YXₐ⎦

    If instead the P=+45deg then this becomes:

    .. code-block::

        ⎡I⎤   ⎡1    0       0    1 ⎤ ⎡XXₐ⎤
        ⎢ ⎥   ⎢                    ⎥ ⎢   ⎥
        ⎢Q⎥   ⎢1    0       0    -1⎥ ⎢XYₐ⎥
        ⎢ ⎥ = ⎢                    ⎥⋅⎢   ⎥
        ⎢U⎥   ⎢0    1       1    0 ⎥ ⎢YXₐ⎥
        ⎢ ⎥   ⎢                    ⎥ ⎢   ⎥
        ⎣V⎦   ⎣0  -1.0⋅ⅈ  1.0⋅ⅈ  0 ⎦ ⎣YYₐ⎦


    or

    .. code-block::

        ⎡I⎤   ⎡    XXₐ + YYₐ     ⎤
        ⎢ ⎥   ⎢                  ⎥
        ⎢Q⎥   ⎢    XXₐ - YYₐ     ⎥
        ⎢ ⎥ = ⎢                  ⎥
        ⎢U⎥   ⎢    XYₐ + YXₐ     ⎥
        ⎢ ⎥   ⎢                  ⎥
        ⎣V⎦   ⎣-ⅈ⋅XYₐ + 1.0⋅ⅈ⋅YXₐ⎦

    However, most imagers (e.g. wsclean, CASA) expect

    .. code-block::

        ⎡I⎤   ⎡0.5    0       0    0.5 ⎤ ⎡XX_w⎤
        ⎢ ⎥   ⎢                        ⎥ ⎢    ⎥
        ⎢Q⎥   ⎢0.5    0       0    -0.5⎥ ⎢XY_w⎥
        ⎢ ⎥ = ⎢                        ⎥⋅⎢    ⎥
        ⎢U⎥   ⎢ 0    0.5     0.5    0  ⎥ ⎢YX_w⎥
        ⎢ ⎥   ⎢                        ⎥ ⎢    ⎥
        ⎣V⎦   ⎣ 0   -0.5⋅i  0.5⋅i   0  ⎦ ⎣YY_w⎦

    or

    .. code-block::

        ⎡I⎤   ⎡  0.5⋅XX_w + 0.5⋅YY_w   ⎤
        ⎢ ⎥   ⎢                        ⎥
        ⎢Q⎥   ⎢  0.5⋅XX_w - 0.5⋅YY_w   ⎥
        ⎢ ⎥ = ⎢                        ⎥
        ⎢U⎥   ⎢  0.5⋅XY_w + 0.5⋅YX_w   ⎥
        ⎢ ⎥   ⎢                        ⎥
        ⎣V⎦   ⎣-0.5⋅i⋅XY_w + 0.5⋅i⋅YX_w⎦

    To convert between the two, we can use the following matrix:

    .. code-block::

        ⎡XX_w⎤   ⎡sin(2.0⋅P) + 1    cos(2.0⋅P)      cos(2.0⋅P)    1 - sin(2.0⋅P)⎤ ⎡XXₐ⎤
        ⎢    ⎥   ⎢                                                              ⎥ ⎢   ⎥
        ⎢XY_w⎥   ⎢ -cos(2.0⋅P)    sin(2.0⋅P) + 1  sin(2.0⋅P) - 1    cos(2.0⋅P)  ⎥ ⎢XYₐ⎥
        ⎢    ⎥ = ⎢                                                              ⎥⋅⎢   ⎥
        ⎢YX_w⎥   ⎢ -cos(2.0⋅P)    sin(2.0⋅P) - 1  sin(2.0⋅P) + 1    cos(2.0⋅P)  ⎥ ⎢YXₐ⎥
        ⎢    ⎥   ⎢                                                              ⎥ ⎢   ⎥
        ⎣YY_w⎦   ⎣1 - sin(2.0⋅P)   -cos(2.0⋅P)     -cos(2.0⋅P)    sin(2.0⋅P) + 1⎦ ⎣YYₐ⎦

    Where `_w` is the 'wsclean' format and _a is the 'ASKAP' format.

    In the case of P=-45deg this becomes:

    .. code-block::

        ⎡XX_w⎤   ⎡0  0   0   2⎤ ⎡XXₐ⎤
        ⎢    ⎥   ⎢            ⎥ ⎢   ⎥
        ⎢XY_w⎥   ⎢0  0   -2  0⎥ ⎢XYₐ⎥
        ⎢    ⎥ = ⎢            ⎥⋅⎢   ⎥
        ⎢YX_w⎥   ⎢0  -2  0   0⎥ ⎢YXₐ⎥
        ⎢    ⎥   ⎢            ⎥ ⎢   ⎥
        ⎣YY_w⎦   ⎣2  0   0   0⎦ ⎣YYₐ⎦

    or

    .. code-block::

        ⎡XX_w⎤   ⎡2⋅YYₐ ⎤
        ⎢    ⎥   ⎢      ⎥
        ⎢XY_w⎥   ⎢-2⋅YXₐ⎥
        ⎢    ⎥ = ⎢      ⎥
        ⎢YX_w⎥   ⎢-2⋅XYₐ⎥
        ⎢    ⎥   ⎢      ⎥
        ⎣YY_w⎦   ⎣2⋅XXₐ ⎦


    And in the case of P=+45deg this becomes:

    .. code-block::

        ⎡XX_w⎤   ⎡2  0  0  0⎤ ⎡XXₐ⎤
        ⎢    ⎥   ⎢          ⎥ ⎢   ⎥
        ⎢XY_w⎥   ⎢0  2  0  0⎥ ⎢XYₐ⎥
        ⎢    ⎥ = ⎢          ⎥⋅⎢   ⎥
        ⎢YX_w⎥   ⎢0  0  2  0⎥ ⎢YXₐ⎥
        ⎢    ⎥   ⎢          ⎥ ⎢   ⎥
        ⎣YY_w⎦   ⎣0  0  0  2⎦ ⎣YYₐ⎦

    or

    .. code-block::

        ⎡XX_w⎤   ⎡2⋅XXₐ⎤
        ⎢    ⎥   ⎢     ⎥
        ⎢XY_w⎥   ⎢2⋅XYₐ⎥
        ⎢    ⎥ = ⎢     ⎥
        ⎢YX_w⎥   ⎢2⋅YXₐ⎥
        ⎢    ⎥   ⎢     ⎥
        ⎣YY_w⎦   ⎣2⋅YYₐ⎦

    That is, when P=+45 H and V are perfectly aligned with X and Y, and
    only the factor of 2 is required.
    """
    theta = pol_axis.to(u.rad).value
    correction_matrix = np.matrix(
        [
            [
                np.sin(2.0 * theta) + 1,
                np.cos(2.0 * theta),
                np.cos(2.0 * theta),
                1 - np.sin(2.0 * theta),
            ],
            [
                -np.cos(2.0 * theta),
                np.sin(2.0 * theta) + 1,
                np.sin(2.0 * theta) - 1,
                np.cos(2.0 * theta),
            ],
            [
                -np.cos(2.0 * theta),
                np.sin(2.0 * theta) - 1,
                np.sin(2.0 * theta) + 1,
                np.cos(2.0 * theta),
            ],
            [
                1 - np.sin(2.0 * theta),
                -np.cos(2.0 * theta),
                -np.cos(2.0 * theta),
                np.sin(2.0 * theta) + 1,
            ],
        ]
    )
    # This is a matrix multiplication broadcasted along the time and channel axes
    rotated_correlations = np.einsum("ij,...j->...i", correction_matrix, correlations)
    if not fix_stokes_factor:
        logger.warning("Not fixing Stokes factor! Multiplying by 0.5...")
        rotated_correlations *= 0.5

    return rotated_correlations


def get_data_chunk(
    ms: Path, chunksize: int, data_column: str = "DATA"
) -> Iterator[np.ndarray]:
    """Generator function that will yield a chunk of data from the `ms` data table.

    Args:
        ms (Path): Measurement set whose data will be iterated over
        chunksize (int): The number of rows to process per chunk
        data_column (str, optional): The column name of the data to iterate. Defaults to "DATA".

    Yields:
        Iterator[np.ndarray]: Chunk of datta to process
    """
    with table(ms.as_posix(), readonly=True, ack=False) as tab:
        data = tab.__getattr__(data_column)
        for i in range(0, len(data), chunksize):
            yield np.array(data[i : i + chunksize])


def get_nchunks(ms: Path, chunksize: int, data_column: str = "DATA") -> int:
    """Returns the number of chunks that are needed to iterator over the datacolumsn
    using a specified `chunksize`.

    Args:
        ms (Path): Measurement sett thatt will be iterated over
        chunksize (int): Size of a single chunk
        data_column (str, optional): Name of the datacolumn that will be iterated over. Defaults to "DATA".

    Returns:
        int: Number of chunks in a measurement set
    """
    with table(ms.as_posix(), readonly=True, ack=False) as tab:
        return int(np.ceil(len(tab.__getattr__(data_column)) / chunksize))


def check_data(ms: Path, data_column: str, corrected_data_column: str) -> bool:
    """Check if the data in the `data_column`, with a correction applied, matches
    the data in the `corrected_data_column`.

    Args:
        ms (Path): MeasurementSet to check
        data_column (str): Data column to check
        corrected_data_column (str): Corrected data column to check

    Returns:
        bool: If the data matches
    """
    with table(ms.as_posix(), readonly=True) as tab:
        idx = 0
        data = tab.__getattr__(data_column).getcell(idx)
        cor_data = tab.__getattr__(corrected_data_column).getcell(idx)
        while (data == 0 + 0j).all():
            idx += 1
            data = tab.__getattr__(data_column).getcell(idx)
            cor_data = tab.__getattr__(corrected_data_column).getcell(idx)

    ang = get_pol_axis(ms)

    return np.allclose(convert_correlations(data, ang), cor_data)


def fix_ms_corrs(
    ms: Path,
    chunksize: int = 10_000,
    data_column: str = "DATA",
    corrected_data_column: str = "CORRECTED_DATA",
    fix_stokes_factor: bool = True,
) -> None:
    """Apply corrections to the ASKAP visibilities to bring them inline with
    what is expectede from other imagers, including CASA and WSClean. The
    original data in `data_column` are copied to `corrected_data_column` before
    the correction is applied. This is done to ensure that the original data
    are not lost.

    If 'corrected_data_column' is detected as an existing column then the
    correction will not be applied.

    Args:
        ms (Path): Path of the ASKAP measurement set tto correct.
        chunksize (int, optional): Size of chunked data to correct. Defaults to 10_000.
        data_column (str, optional): The name of the data column to correct. Defaults to "DATA".
        corrected_data_column (str, optional): The name of the corrected data column. Defaults to "CORRECTED_DATA".
        fix_stokes_factor (bool, optional): Whether to fix the Stokes factor. Defaults to True.
    """

    _function_args = [f"{key}={val}" for key, val in locals().items()]

    logger.info(
        f"Correcting {data_column} of {str(ms)}.", ms=ms, app_params=_function_args
    )
    # Do checks
    with table(ms.as_posix(), readonly=True, ack=False) as tab:
        cols = tab.colnames()
        # Check if 'data_column' exists
        if data_column not in cols:
            logger.critical(
                f"Column {data_column} does not exist in {ms}! Exiting...",
                ms=ms,
                app_params=_function_args,
            )
            return
        # Check if 'corrected_data_column' exists
        if corrected_data_column in cols:
            logger.critical(
                f"Column {corrected_data_column} already exists in {ms}! Exiting...",
                ms=ms,
                app_params=_function_args,
            )
            if check_data(ms, data_column, corrected_data_column):
                logger.critical(
                    f"We checked the data in {data_column} against {corrected_data_column} and it looks like the correction has already been applied!",
                    ms=ms,
                    app_params=_function_args,
                )
            return

        feed1 = np.unique(tab.getcol("FEED1"))
        feed2 = np.unique(tab.getcol("FEED2"))

        # For some ASKAP observations the orientation of the third-axis changes
        # throughout the observation. For example, bandpass observations vary
        # this direction as each beam cycles in the footprint cycles over the
        # calibrator source.
        assert (
            len(feed1) == 1 and len(feed2) == 1
        ), "Found more than one feed orientation!"
        assert (
            feed1[0] == feed2[0]
        ), f"The unique feed enteries available in the data table differ, {feed1=} {feed2=}"

        # The two assertions above should enfore enough constraint
        # to make sure the rotation matix constructed is correct
        feed_idx = feed1[0]

    # Get the polarization axis
    pol_axis = get_pol_axis(ms, feed_idx=feed_idx)
    logger.info(f"Polarization axis is {pol_axis}", ms=ms, app_params=_function_args)

    # Check for pol_axis = 0deg - no correction needed
    if pol_axis == 0 * u.deg and not fix_stokes_factor:
        logger.critical(
            f"Pol axis is 0deg. No correction needed. Exiting...",
            ms=ms,
            app_params=_function_args,
        )
        return

    # Get the data chunk by chunk and convert the correlations
    # then write them back to the MS in the 'data_column' column
    data_chunks = get_data_chunk(ms, chunksize, data_column=data_column)
    nchunks = get_nchunks(ms, chunksize, data_column=data_column)
    start_row = 0
    with table(ms.as_posix(), readonly=False, ack=False) as tab:
        desc = makecoldesc(data_column, tab.getcoldesc(data_column))
        desc["name"] = corrected_data_column
        try:
            tab.addcols(desc)
        except RuntimeError:
            # This can happen if this correction/script has been run more
            # than once on the same measurement set.
            # Putting this here for interactive use when you might muck around with the MS
            logger.critical(
                (
                    f"Column {corrected_data_column} already exists in {ms}! You should never see this message! "
                    f"Possible an existing {data_column} has already been corrected. "
                    f"No correction will be applied. "
                ),
                ms=ms,
                app_params=_function_args,
            )
        else:
            # Only perform this correction if the data column was
            # successfully renamed.
            for data_chunk in tqdm(
                data_chunks, total=nchunks, file=TQDM_OUT, desc="Rotating correlations"
            ):
                data_chunk_cor = convert_correlations(
                    data_chunk,
                    pol_axis,
                    fix_stokes_factor=fix_stokes_factor,
                )
                tab.putcol(
                    corrected_data_column,
                    data_chunk_cor,
                    startrow=start_row,
                    nrow=len(data_chunk_cor),
                )
                tab.flush()
                start_row += len(data_chunk_cor)

    logger.info(
        f"Finished correcting {data_column} of {str(ms)}. Written to {corrected_data_column} column.",
        ms=ms,
        app_params=_function_args,
    )
    logger.info(
        f"Updating the FEED table by a rotation of {pol_axis}",
        ms=ms,
        app_params=_function_args,
    )
    set_pol_axis(ms, pol_axis, feed_idx=feed_idx)
    logger.info(
        f"Updated the RECEPTOR_ANGLE column with a rotation of {pol_axis}",
        ms=ms.as_posix(),
        app_params=_function_args,
    )

    logger.info("Done!")


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ms", type=Path, help="The MS to fix")
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
    parser.add_argument(
        "--no-fix-stokes-factor",
        dest="no_fix_stokes_factor",
        action="store_true",
        help="Don't fix the Stokes factor. Use this if you have *not* used ASKAPsoft. If you have used ASKAPsoft, you should leave this option alone.",
    )
    args = parser.parse_args()
    fix_ms_corrs(
        Path(args.ms),
        chunksize=args.chunksize,
        data_column=args.data_column,
        corrected_data_column=args.corrected_data_column,
        fix_stokes_factor=not args.no_fix_stokes_factor,
    )


if __name__ == "__main__":
    cli()
