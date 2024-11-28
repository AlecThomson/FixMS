#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the fixing the MS correlations
"""

import importlib.resources as resources
import logging
import shutil
from pathlib import Path
from typing import NamedTuple

import astropy.units as u
import numpy as np
import pytest
from casacore.tables import table

from fixms.fix_ms_corrs import (
    check_data,
    convert_correlations,
    fix_ms_corrs,
    get_pol_axis,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Mueller(NamedTuple):
    stokes_I: np.ndarray
    stokes_Q: np.ndarray
    stokes_U: np.ndarray
    stokes_V: np.ndarray


class ExampleData(NamedTuple):
    original_ms_path: Path
    fixed_ms_path: Path
    data_column: str
    corrected_data_column: str
    pol_axis: u.Quantity
    original_data: np.ndarray
    fixed_data: np.ndarray
    fixed_corrected_data: np.ndarray


def get_packaged_resource_path(package: str, filename: str) -> Path:
    """Load in the path of a package sources.

    The `package` argument is passed as a though the module
    is being specified as an import statement.

    Args:
        package (str): The module path to the resources
        filename (str): Filename of the datafile to load

    Returns:
        Path: The absolute path to the packaged resource file
    """
    logger.info(f"Loading {package=} for {filename=}")

    dir_path = resources.files(package)
    full_path = dir_path / filename

    logger.debug(f"Resolved {full_path=}")

    return Path(full_path)


@pytest.fixture
def ms_rotated_example(tmpdir) -> ExampleData:
    ms_zip = Path(
        get_packaged_resource_path(
            package="fixms.data",
            filename="RACS_1313-72.SB57526.split.ms.zip",
        )
    )
    outpath = Path(tmpdir) / "standard"

    shutil.unpack_archive(ms_zip, outpath)

    original_ms_path = Path(outpath) / "RACS_1313-72.SB57526.split.ms"
    fixed_ms_path = Path(outpath) / "RACS_1313-72.SB57526.split.fixed.ms"
    shutil.copytree(original_ms_path, fixed_ms_path)

    data_column = "DATA"
    corrected_data_column = "CORRECTED_DATA"
    # From OMP - common.target.src1.pol_axis = [pa_fixed, 115.859]
    pol_axis = 115.859 * u.deg

    # Run the fix
    logger.info(f"Running fix_ms_corrs on {fixed_ms_path}")
    fix_ms_corrs(
        fixed_ms_path,
        data_column=data_column,
        corrected_data_column=corrected_data_column,
    )
    with table(original_ms_path.as_posix(), readonly=True) as tab:
        original_data = tab.getcol(data_column)

    with table(fixed_ms_path.as_posix(), readonly=True) as tab:
        fixed_data = tab.getcol(data_column)
        fixed_corrected_data = tab.getcol(corrected_data_column)

    logger.debug(f"{fixed_data.shape=}")

    yield ExampleData(
        original_ms_path=original_ms_path,
        fixed_ms_path=fixed_ms_path,
        data_column=data_column,
        corrected_data_column=corrected_data_column,
        pol_axis=pol_axis,
        original_data=original_data,
        fixed_data=fixed_data,
        fixed_corrected_data=fixed_corrected_data,
    )

    for f in (original_ms_path, fixed_ms_path):
        if f.exists():
            shutil.rmtree(f)


@pytest.fixture
def ms_standard_example(tmpdir) -> ExampleData:
    ms_zip = Path(
        get_packaged_resource_path(
            package="fixms.data",
            filename="SB60933.RACS_1727+37.split.ms.zip",
        )
    )
    outpath = Path(tmpdir) / "standard"

    shutil.unpack_archive(ms_zip, outpath)

    original_ms_path = Path(outpath) / "SB60933.RACS_1727+37.split.ms"
    fixed_ms_path = Path(outpath) / "SB60933.RACS_1727+37.split.fixed.ms"
    shutil.copytree(original_ms_path, fixed_ms_path)

    data_column = "DATA"
    corrected_data_column = "CORRECTED_DATA"
    # From OMP - common.target.src1.pol_axis = [pa_fixed, -45]
    pol_axis = -45 * u.deg

    # Run the fix
    logger.info(f"Running fix_ms_corrs on {fixed_ms_path}")
    fix_ms_corrs(
        fixed_ms_path,
        data_column=data_column,
        corrected_data_column=corrected_data_column,
    )

    with table(original_ms_path.as_posix(), readonly=True) as tab:
        original_data = tab.getcol(data_column)

    with table(fixed_ms_path.as_posix(), readonly=True) as tab:
        fixed_data = tab.getcol(data_column)
        fixed_corrected_data = tab.getcol(corrected_data_column)

    logger.debug(f"{fixed_data.shape=}")

    yield ExampleData(
        original_ms_path=original_ms_path,
        fixed_ms_path=fixed_ms_path,
        data_column=data_column,
        corrected_data_column=corrected_data_column,
        pol_axis=pol_axis,
        original_data=original_data,
        fixed_data=fixed_data,
        fixed_corrected_data=fixed_corrected_data,
    )

    for f in (original_ms_path, fixed_ms_path):
        if f.exists():
            shutil.rmtree(f)


def test_get_pol_axis(ms_standard_example, ms_rotated_example):
    # Test that the pol axis is correct
    for ms in (ms_standard_example, ms_rotated_example):
        pol_axis_original = get_pol_axis(ms.original_ms_path, col="RECEPTOR_ANGLE")
        pol_axis_fixed = get_pol_axis(ms.fixed_ms_path, col="INSTRUMENT_RECEPTOR_ANGLE")
        rot_pol_axis_fixed = get_pol_axis(ms.fixed_ms_path, col="RECEPTOR_ANGLE")

        assert np.isclose(
            pol_axis_original, ms.pol_axis
        ), f"Pol axis is incorrect {pol_axis_original=}"
        assert np.isclose(
            pol_axis_fixed, ms.pol_axis
        ), f"Pol axis is incorrect {pol_axis_fixed=}"
        assert np.isclose(
            rot_pol_axis_fixed, 0 * u.deg
        ), f"Pol axis is incorrect {rot_pol_axis_fixed=}"


def test_column_exists(ms_standard_example, ms_rotated_example):
    # Check that CORRECTED_DATA is on disk
    for ms in (ms_standard_example, ms_rotated_example):
        with table(ms.fixed_ms_path.as_posix()) as tab:
            assert (
                ms.corrected_data_column in tab.colnames()
            ), f"{ms.corrected_data_column} not in MS"


def test_original_data(ms_standard_example, ms_rotated_example):
    # Check that the read-only data is unchanged
    for ms in (ms_standard_example, ms_rotated_example):
        with table(ms.fixed_ms_path.as_posix(), readonly=True) as tab:
            fixed_data = tab.getcol(ms.data_column)

        assert np.allclose(ms.original_data, fixed_data), f"{ms.data_column} changed"


def test_corrected_data(ms_standard_example, ms_rotated_example):
    for ms in (ms_standard_example, ms_rotated_example):
        # Get the receptor angle
        ang = get_pol_axis(ms.original_ms_path)
        # Check the conversion as written to disk
        assert np.allclose(
            convert_correlations(ms.original_data, ang),
            ms.fixed_corrected_data,
        ), "Data write failed"


def test_check_data(ms_standard_example, ms_rotated_example):
    for ms in (ms_standard_example, ms_rotated_example):
        assert not check_data(
            ms.fixed_ms_path, ms.data_column, ms.corrected_data_column
        ), f"Corrected data column is incorrect in {ms.fixed_ms_path.name}"


def askap_stokes_mat(ms):
    theta = get_pol_axis(ms.original_ms_path)
    xx, xy, yx, yy = ms.original_data.T
    correlations = np.array([xx, xy, yx, yy])
    logger.debug(f"{correlations.shape=}")
    # Equation D.1 from ASKAP Observation Guide
    rot = np.array(
        [
            [1, 0, 0, 1],
            [
                np.sin(2 * theta),
                np.cos(2 * theta),
                np.cos(2 * theta),
                -np.sin(2 * theta),
            ],
            [
                -np.cos(2 * theta),
                np.sin(2 * theta),
                np.sin(2 * theta),
                np.cos(2 * theta),
            ],
            [0, -1j, 1j, 0],
        ]
    )
    logger.debug(f"{rot.shape=}")
    # I, Q, U, V = rot @ correlations
    stokes_I, stokes_Q, stokes_U, stokes_V = np.einsum(
        "ij,j...->i...", rot, correlations
    )
    return Mueller(stokes_I, stokes_Q, stokes_U, stokes_V)


def askap_stokes(ms):
    theta = get_pol_axis(ms.original_ms_path)
    xx, xy, yx, yy = ms.original_data.T
    assert theta == -45 * u.deg, "Only works for theta = -45 deg, got {theta=}"
    stokes_I = xx + yy
    stokes_Q = yy - xx
    stokes_U = -(xy + yx)
    stokes_V = 1j * (yx - xy)

    return Mueller(stokes_I, stokes_Q, stokes_U, stokes_V)


def get_wsclean_stokes(ms):
    # from https://gitlab.com/aroffringa/aocommon/-/blob/master/include/aocommon/polarization.h
    # *   XX = I + Q  ;   I = (XX + YY)/2
    # *   XY = U + iV ;   Q = (XX - YY)/2
    # *   YX = U - iV ;   U = (XY + YX)/2
    # *   YY = I - Q  ;   V = -i(XY - YX)/2
    xx, xy, yx, yy = ms.fixed_corrected_data.T
    stokes_I = 0.5 * (xx + yy)
    stokes_Q = 0.5 * (xx - yy)
    stokes_U = 0.5 * (xy + yx)
    stokes_V = -0.5j * (xy - yx)

    return Mueller(stokes_I, stokes_Q, stokes_U, stokes_V)


def test_rotated_data_I(ms_standard_example):
    mueller_a = askap_stokes(ms_standard_example)
    mueller_a_mat = askap_stokes_mat(ms_standard_example)
    assert np.allclose(
        mueller_a.stokes_I, mueller_a_mat.stokes_I, atol=1e-4
    ), "Stokes rotation I failed"


def test_rotated_data_Q(ms_standard_example):
    mueller_a = askap_stokes(ms_standard_example)
    mueller_a_mat = askap_stokes_mat(ms_standard_example)
    assert np.allclose(
        mueller_a.stokes_Q, mueller_a_mat.stokes_Q, atol=1e-4
    ), f"Stokes rotation Q failed for {ms_standard_example.fixed_ms_path.name}"


def test_rotated_data_U(ms_standard_example):
    mueller_a = askap_stokes(ms_standard_example)
    mueller_a_mat = askap_stokes_mat(ms_standard_example)
    assert np.allclose(
        mueller_a.stokes_U, mueller_a_mat.stokes_U, atol=1e-4
    ), f"Stokes rotation U failed for {ms_standard_example.fixed_ms_path.name}"


def test_rotated_data_V(ms_standard_example):
    mueller_a = askap_stokes(ms_standard_example)
    mueller_a_mat = askap_stokes_mat(ms_standard_example)
    assert np.allclose(
        mueller_a.stokes_V, mueller_a_mat.stokes_V, atol=1e-4
    ), f"Stokes rotation V failed for {ms_standard_example.fixed_ms_path.name}"


def test_stokes_I(ms_standard_example, ms_rotated_example):
    for ms in (ms_standard_example, ms_rotated_example):
        mueller_a = askap_stokes_mat(ms)
        mueller_w = get_wsclean_stokes(ms)
        assert np.allclose(
            mueller_a.stokes_I, mueller_w.stokes_I, atol=1e-4
        ), f"ASKAP and WSClean disagree on Stokes I in {ms.fixed_ms_path.name}"


def test_stokes_Q(ms_standard_example, ms_rotated_example):
    for ms in (ms_standard_example, ms_rotated_example):
        mueller_a = askap_stokes_mat(ms)
        mueller_w = get_wsclean_stokes(ms)
        assert np.allclose(
            mueller_a.stokes_Q, mueller_w.stokes_Q, atol=1e-1
        ), f"ASKAP and WSClean disagree on Stokes Q in {ms.fixed_ms_path.name}"


def test_stokes_U(ms_standard_example, ms_rotated_example):
    for ms in (ms_standard_example, ms_rotated_example):
        mueller_a = askap_stokes_mat(ms)
        mueller_w = get_wsclean_stokes(ms)
        assert np.allclose(
            mueller_a.stokes_U, mueller_w.stokes_U, atol=1e-4
        ), f"ASKAP and WSClean disagree on Stokes U in {ms.fixed_ms_path.name}"


def test_stokes_V(ms_standard_example, ms_rotated_example):
    for ms in (ms_standard_example, ms_rotated_example):
        mueller_a = askap_stokes_mat(ms)
        mueller_w = get_wsclean_stokes(ms)
        assert np.allclose(
            mueller_a.stokes_V, mueller_w.stokes_V, atol=1e-4
        ), f"ASKAP and WSClean disagree on Stokes V in {ms.fixed_ms_path.name}"
