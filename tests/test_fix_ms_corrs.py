#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the fixing the MS correlations
"""
import logging
import shutil
import unittest
from pathlib import Path
from typing import NamedTuple

import astropy.units as u
import numpy as np
from casacore.tables import table

from fixms.fix_ms_corrs import convert_correlations, fix_ms_corrs, get_pol_axis

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Mueller(NamedTuple):
    I: np.ndarray
    Q: np.ndarray
    U: np.ndarray
    V: np.ndarray


class Tester(unittest.TestCase):
    # def __init__(self):
    def setUp(self) -> None:
        # Copy the read-only MS file to a temporary file
        self.read_only_ms = Path(
            "tests/scienceData.RACS_0012+00.SB45305.RACS_0012+00.beam00_averaged_cal.leakage.ms"
        )
        self.ms = Path("tests/test.ms")
        # Copy the read-only MS file to a temporary file
        logger.info(f"Copying {self.read_only_ms} to {self.ms}")
        shutil.copytree(self.read_only_ms, self.ms, dirs_exist_ok=True)
        # Allow writing to the MS file
        self.data_column = "DATA"
        self.corrected_data_column = "CORRECTED_DATA"
        self.pol_axis = -45 * u.deg

        # Run the fix
        logger.info(f"Running fix_ms_corrs on {self.ms}")
        fix_ms_corrs(
            self.ms,
            data_column=self.data_column,
            corrected_data_column=self.corrected_data_column,
        )

        # Find first row with good data
        with table(self.ms.as_posix(), readonly=True) as tab:
            idx = 0
            data = tab.__getattr__(self.data_column).getcell(idx)
            cor_data = tab.__getattr__(self.corrected_data_column).getcell(idx)
            while (data == 0 + 0j).all():
                idx += 1
                data = tab.__getattr__(self.data_column).getcell(idx)
                cor_data = tab.__getattr__(self.corrected_data_column).getcell(idx)

        self.data = data
        self.cor_data = cor_data

        # Get the read-only data
        with table(self.read_only_ms.as_posix(), readonly=True) as tab:
            data = tab.__getattr__(self.data_column).getcell(idx)
        self.read_only_data = data

    def test_get_pol_axis(self):
        # Test that the pol axis is correct
        pol_axis = get_pol_axis(self.ms)
        assert pol_axis == self.pol_axis

    def test_column_exists(self):
        # Check that CORRECTED_DATA is on disk
        with table(self.ms.as_posix()) as tab:
            assert (
                self.corrected_data_column in tab.colnames()
            ), f"{self.corrected_data_column} not in MS"

    def test_read_only_data(self):
        # Check that the read-only data is unchanged
        assert np.allclose(
            self.read_only_data, self.data
        ), f"{self.data_column} changed"

    def test_corrected_data(self):
        # Get the receptor angle
        ang = get_pol_axis(self.ms)

        # Check the conversion as written to disk
        assert np.allclose(
            convert_correlations(self.data, ang), self.cor_data
        ), "Data write failed"

    def askap_stokes_mat(self):
        theta = get_pol_axis(self.ms) + 45 * u.deg
        xx, xy, yx, yy = self.data.T
        correlations = np.array([xx, xy, yx, yy])
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
        I, Q, U, V = rot @ correlations
        return Mueller(I, Q, U, V)

    def askap_stokes(self):
        theta = get_pol_axis(self.ms) + 45 * u.deg
        xx, xy, yx, yy = self.data.T
        assert theta == 0 * u.deg, "Only works for theta = 0 deg"
        I = xx + yy
        Q = xy + yx
        U = yy - xx
        V = 1j * (yx - xy)

        return Mueller(I, Q, U, V)

    def get_wsclean_stokes(self):
        # from https://gitlab.com/aroffringa/aocommon/-/blob/master/include/aocommon/polarization.h
        # *   XX = I + Q  ;   I = (XX + YY)/2
        # *   XY = U + iV ;   Q = (XX - YY)/2
        # *   YX = U - iV ;   U = (XY + YX)/2
        # *   YY = I - Q  ;   V = -i(XY - YX)/2
        xx, xy, yx, yy = self.cor_data.T
        I = 0.5 * (xx + yy)
        Q = 0.5 * (xx - yy)
        U = 0.5 * (xy + yx)
        V = -0.5j * (xy - yx)

        return Mueller(I, Q, U, V)

    def test_rotated_data_I(self):
        mueller_a = self.askap_stokes()
        mueller_a_mat = self.askap_stokes_mat()
        assert np.allclose(
            mueller_a.I, mueller_a_mat.I, atol=1e-4
        ), "Stokes rotation I failed"

    def test_rotated_data_Q(self):
        mueller_a = self.askap_stokes()
        mueller_a_mat = self.askap_stokes_mat()
        assert np.allclose(
            mueller_a.Q, mueller_a_mat.Q, atol=1e-4
        ), "Stokes rotation Q failed"

    def test_rotated_data_U(self):
        mueller_a = self.askap_stokes()
        mueller_a_mat = self.askap_stokes_mat()
        assert np.allclose(
            mueller_a.U, mueller_a_mat.U, atol=1e-4
        ), "Stokes rotation U failed"

    def test_rotated_data_V(self):
        mueller_a = self.askap_stokes()
        mueller_a_mat = self.askap_stokes_mat()
        assert np.allclose(
            mueller_a.V, mueller_a_mat.V, atol=1e-4
        ), "Stokes rotation V failed"

    def test_stokes_I(self):
        mueller_a = self.askap_stokes_mat()
        mueller_w = self.get_wsclean_stokes()
        assert np.allclose(
            mueller_a.I, mueller_w.I, atol=1e-4
        ), "ASKAP and WSClean disagree on Stokes I"

    def test_stokes_Q(self):
        mueller_a = self.askap_stokes_mat()
        mueller_w = self.get_wsclean_stokes()
        assert np.allclose(
            mueller_a.Q, mueller_w.Q, atol=1e-2
        ), "ASKAP and WSClean disagree on Stokes Q"

    def test_stokes_U(self):
        mueller_a = self.askap_stokes_mat()
        mueller_w = self.get_wsclean_stokes()
        assert np.allclose(
            mueller_a.U, mueller_w.U, atol=1e-4
        ), "ASKAP and WSClean disagree on Stokes U"

    def test_stokes_V(self):
        mueller_a = self.askap_stokes_mat()
        mueller_w = self.get_wsclean_stokes()
        assert np.allclose(
            mueller_a.V, mueller_w.V, atol=1e-4
        ), "ASKAP and WSClean disagree on Stokes V"

    def tearDown(self) -> None:
        # Remove the temporary MS file
        logger.info(f"Removing {self.ms}")
        shutil.rmtree(self.ms)
