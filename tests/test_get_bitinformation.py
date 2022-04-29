#!/usr/bin/env python

"""Tests for `xbitinfo` package."""
import os

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_equal

import xbitinfo as xb


def bitinfo_assert_different(bitinfo1, bitinfo2):
    """Fail bitinfo different values."""
    assert (bitinfo1 != bitinfo2).any()



def test_get_bitinformation_returns_xr_Dataset():
    """Test xb.get_bitinformation returns xr.Dataset."""
    ds = xr.tutorial.load_dataset("rasm")
    assert isinstance(xb.get_bitinformation(ds, axis=0), xr.Dataset)


def test_get_bitinformation_dim():
    """Test xb.get_bitinformation is sensitive to dim."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo0 = xb.get_bitinformation(ds, axis=0)
    bitinfo2 = xb.get_bitinformation(ds, axis=2)
    bitinfo_assert_different(bitinfo0, bitinfo2)


def test_get_bitinformation_dim_string_equals_axis_int():
    """Test xb.get_bitinformation undestands xarray dimension names the same way as axis as integers."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfox = xb.get_bitinformation(ds, dim="x")
    bitinfo2 = xb.get_bitinformation(ds, axis=2)
    assert_equal(bitinfox, bitinfo2)


def test_get_bitinformation_masked_value():
    """Test xb.get_bitinformation is sensitive to masked_value."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo = xb.get_bitinformation(ds, dim="x")
    bitinfo_no_mask = xb.get_bitinformation(ds, dim="x", masked_value="nothing")
    bitinfo_no_mask_None = xb.get_bitinformation(ds, dim="x", masked_value=None)
    assert_equal(bitinfo_no_mask, bitinfo_no_mask_None)
    bitinfo_assert_different(bitinfo, bitinfo_no_mask)


def test_get_bitinformation_set_zero_insignificant():
    """Test xb.get_bitinformation is sensitive to set_zero_insignificant."""
    ds = xr.tutorial.load_dataset("air_temperature")
    dim = "lon"
    bitinfo_szi_False = xb.get_bitinformation(ds, dim=dim, set_zero_insignificant=False)
    bitinfo_szi_True = xb.get_bitinformation(ds, dim=dim, set_zero_insignificant=True)
    bitinfo = xb.get_bitinformation(ds, dim=dim)
    bitinfo_assert_different(bitinfo, bitinfo_szi_False)
    assert_equal(bitinfo, bitinfo_szi_True)


def test_get_bitinformation_confidence():
    """Test xb.get_bitinformation is sensitive to confidence."""
    ds = xr.tutorial.load_dataset("air_temperature")
    dim = "lon"
    bitinfo_conf99 = xb.get_bitinformation(ds, dim=dim, confidence=0.99)
    bitinfo_conf50 = xb.get_bitinformation(ds, dim=dim, confidence=0.5)
    bitinfo = xb.get_bitinformation(ds, dim=dim)
    bitinfo_assert_different(bitinfo_conf99, bitinfo_conf50)
    assert_equal(bitinfo, bitinfo_conf99)


def test_get_bitinformation_label(rasm):
    """Test xb.get_bitinformation serializes when label given."""
    ds = rasm
    xb.get_bitinformation(ds, dim="x", label="rasm")
    assert os.path.exists("rasm.json")
    # second call should be faster
    xb.get_bitinformation(ds, dim="x", label="rasm")
    os.remove("rasm.json")


@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_get_bitinformation_dtype(rasm, dtype):
    """Test xb.get_bitinformation returns correct number of bits depending on dtype."""
    ds = rasm.astype(dtype)
    v = list(ds.data_vars)[0]
    assert len(xb.get_bitinformation(ds, dim="x")[v]) == int(dtype.replace("float", ""))
