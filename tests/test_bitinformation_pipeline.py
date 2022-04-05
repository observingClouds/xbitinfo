#!/usr/bin/env python

"""Tests for `bitinformation_pipeline` package."""

import xarray as xr

import bitinformation_pipeline as bp


def test_get_bitinformation():
    """Test bm.get_bitinformation."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo = bp.get_bitinformation(ds, dim=1)
    print(bitinfo)
    assert bitinfo
