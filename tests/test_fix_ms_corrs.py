#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the fixing the MS correlations
"""
from pathlib import Path
import shutil
import unittest
import logging
from functools import cached_property
from typing import NamedTuple

from casacore.tables import table
import numpy as np
import astropy.units as u
import pytest
from fixms.fix_ms_corrs import (
    get_pol_axis, convert_correlations, fix_ms_corrs
)

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
        self.read_only_ms = Path('scienceData_SB13671_RACS_1237+00A.beam00_averaged_cal.ms')
        self.ms = Path('test.ms')
        # Copy the read-only MS file to a temporary file
        logger.info(f"Copying {self.read_only_ms} to {self.ms}")
        shutil.copytree(self.read_only_ms, self.ms, dirs_exist_ok=True)
        # Allow writing to the MS file
        self.data_column = "DATA"
        self.corrected_data_column = "CORRECTED_DATA"
        self.pol_axis = 45*u.deg

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
            while (data == 0+0j).all():
                idx +=1
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
            assert self.corrected_data_column in tab.colnames(), f"{self.corrected_data_column} not in MS"

    def test_read_only_data(self):
        # Check that the read-only data is unchanged
        assert np.allclose(self.read_only_data, self.data), f"{self.data_column} changed"

    def test_corrected_data(self):
        # Get the receptor angle
        ang = get_pol_axis(self.ms)

        # Check the conversion as written to disk
        assert np.allclose(convert_correlations(self.data,ang), self.cor_data), "Data write failed"

    def get_muellers(self):
        ang = get_pol_axis(self.ms)
        xx_a, xy_a, yx_a, yy_a = self.data.T
        xx_w, xy_w, yx_w, yy_w = self.cor_data.T

        if ang == 45*u.deg:
            I_a = xx_a + yy_a
            Q_a = xy_a + yx_a
            U_a = yy_a - xx_a
            V_a = 1j*(yx_a - xy_a)

            I_w = 0.5*(xx_w + yy_w)
            Q_w = 0.5*(xx_w - yy_w)
            U_w = 0.5*(xy_w + yx_w)
            V_w = -0.5j*(xy_w - yx_w)

        else:
            raise NotImplementedError(f"Angle {ang} not implemented")

        mueller_a = Mueller(I_a, Q_a, U_a, V_a)
        mueller_w = Mueller(I_w, Q_w, U_w, V_w)
        return mueller_a, mueller_w

    def test_stokes_I(self):
        mueller_a, mueller_w = self.get_muellers()
        assert np.allclose(mueller_a.I, mueller_w.I, atol=1e-4), "Stokes I changed"

    def test_stokes_Q(self):
        mueller_a, mueller_w = self.get_muellers()
        assert np.allclose(mueller_a.Q, mueller_w.Q, atol=1e-4), "Stokes Q changed"
    
    def test_stokes_U(self):
        mueller_a, mueller_w = self.get_muellers()
        assert np.allclose(mueller_a.U, mueller_w.U, atol=1e-4), "Stokes U changed"

    def test_stokes_V(self):
        mueller_a, mueller_w = self.get_muellers()
        assert np.allclose(mueller_a.V, mueller_w.V, atol=1e-4), "Stokes V changed"
                

    def tearDown(self) -> None:
        # Remove the temporary MS file
        logger.info(f"Removing {self.ms}")
        shutil.rmtree(self.ms)