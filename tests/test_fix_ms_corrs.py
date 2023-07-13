#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the fixing the MS correlations
"""
import logging
import shutil
import unittest
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

import astropy.units as u
import numpy as np
import pytest
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
            "scienceData.RACS_0012+00.SB45305.RACS_0012+00.beam00_averaged_cal.leakage.ms"
        )
        self.ms = Path("test.ms")
        # Copy the read-only MS file to a temporary file
        logger.info(f"Copying {self.read_only_ms} to {self.ms}")
        shutil.copytree(self.read_only_ms, self.ms, dirs_exist_ok=True)
        # Allow writing to the MS file
        self.data_column = "DATA"
        self.corrected_data_column = "CORRECTED_DATA"
        self.pol_axis = 45 * u.deg

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

    @staticmethod
    def askap_stokes_mat(xx, xy, yx, yy, theta):
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

    @staticmethod
    def askap_stokes(xx, xy, yx, yy, theta):
        assert theta == -45 * u.deg, "Only works for theta = -45 deg"
        I = xx + yy
        Q = xy + yx
        U = yy - xx
        V = 1j * (yx - xy)

        return Mueller(I, Q, U, V)

    def get_muellers(self):
        ang = get_pol_axis(self.ms)
        xx_a, xy_a, yx_a, yy_a = self.data.T
        xx_w, xy_w, yx_w, yy_w = self.cor_data.T

        mueller_a = self.askap_stokes(xx_a, xy_a, yx_a, yy_a, ang)

        # from https://gitlab.com/aroffringa/aocommon/-/blob/master/include/aocommon/polarization.h
        # *   XX = I + Q  ;   I = (XX + YY)/2
        # *   XY = U + iV ;   Q = (XX - YY)/2
        # *   YX = U - iV ;   U = (XY + YX)/2
        # *   YY = I - Q  ;   V = -i(XY - YX)/2

        I_w = 0.5 * (xx_w + yy_w)
        Q_w = 0.5 * (xx_w - yy_w)
        U_w = 0.5 * (xy_w + yx_w)
        V_w = -0.5j * (xy_w - yx_w)

        mueller_w = Mueller(I_w, Q_w, U_w, V_w)
        return mueller_a, mueller_w

    def test_stokes_I(self):
        mueller_a, mueller_w = self.get_muellers()
        assert np.allclose(mueller_a.I, mueller_w.I, atol=1e-4), "Stokes I failed"

    def test_stokes_Q(self):
        mueller_a, mueller_w = self.get_muellers()
        assert np.allclose(mueller_a.Q, mueller_w.Q, atol=1e-4), "Stokes Q failed"

    def test_stokes_U(self):
        mueller_a, mueller_w = self.get_muellers()
        assert np.allclose(mueller_a.U, mueller_w.U, atol=1e-4), "Stokes U failed"

    def test_stokes_V(self):
        mueller_a, mueller_w = self.get_muellers()
        assert np.allclose(mueller_a.V, mueller_w.V, atol=1e-4), "Stokes V failed"

    def tearDown(self) -> None:
        # Remove the temporary MS file
        logger.info(f"Removing {self.ms}")
        shutil.rmtree(self.ms)
