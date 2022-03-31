#!/usr/bin/env python

"""Tests for `bitinformation_pipeline` package."""

import pytest

from bitinformation_pipeline import bitinformation_pipeline as bm
import xarray as xr
    

def test_get_bitinformation():
    """Test bm.get_bitinformation."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo = bm.get_bitinformation(ds, dim=1)
