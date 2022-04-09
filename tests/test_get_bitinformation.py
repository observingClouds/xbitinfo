#!/usr/bin/env python

"""Tests for `bitinformation_pipeline` package."""
import os

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_equal

import bitinformation_pipeline as bp


def bitinfo_assert_equal(bitinfo1, bitinfo2):
    assert list(bitinfo1.keys()) == list(bitinfo2.keys()), print(
        f"lhs = {bitinfo1.keys()} vs rhs = {bitinfo2.keys()}"
    )
    for v in bitinfo1.keys():
        assert_equal(bitinfo1[v], bitinfo2[v])


def bitinfo_assert_allclose(bitinfo1, bitinfo2):
    assert list(bitinfo1.keys()) == list(bitinfo2.keys()), print(
        f"lhs = {bitinfo1.keys()} vs rhs = {bitinfo2.keys()}"
    )
    for v in bitinfo1.keys():
        assert_allclose(bitinfo1[v], bitinfo2[v])


def bitinfo_assert_different(bitinfo1, bitinfo2):
    """Fail when identical in all keys and values."""
    bitinfo1_values = np.array([i for i in bitinfo1.values()])
    bitinfo2_values = np.array([i for i in bitinfo2.values()])
    assert not (bitinfo1_values == bitinfo2_values).all()


def test_get_bitinformation():
    """Test bp.get_bitinformation."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo = bp.get_bitinformation(ds, dim=1)
    print(bitinfo)
    assert bitinfo


def test_get_bitinformation_dim():
    """Test bp.get_bitinformation is sensitive to dim."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo1 = bp.get_bitinformation(ds, dim=1)
    bitinfo3 = bp.get_bitinformation(ds, dim=3)
    bitinfo_assert_different(bitinfo1, bitinfo3)


def test_get_bitinformation_dim_string():
    """Test bp.get_bitinformation undestands xarray dimension names."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfox = bp.get_bitinformation(ds, dim="x")
    bitinfo3 = bp.get_bitinformation(ds, dim=3)
    bitinfo_assert_equal(bitinfox, bitinfo3)


def test_get_bitinformation_masked_value():
    """Test bp.get_bitinformation is sensitive to masked_value."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo = bp.get_bitinformation(ds, dim="x")
    bitinfo_no_mask = bp.get_bitinformation(ds, dim="x", masked_value="nothing")
    bitinfo_assert_different(bitinfo, bitinfo_no_mask)


def test_get_bitinformation_set_zero_insignificant():
    """Test bp.get_bitinformation is sensitive to set_zero_insignificant."""
    ds = xr.tutorial.load_dataset("air_temperature")
    dim = "lon"
    bitinfo_szi_False = bp.get_bitinformation(ds, dim=dim, set_zero_insignificant=False)
    bitinfo_szi_True = bp.get_bitinformation(ds, dim=dim, set_zero_insignificant=True)
    bitinfo = bp.get_bitinformation(ds, dim=dim)
    bitinfo_assert_different(bitinfo, bitinfo_szi_False)
    bitinfo_assert_equal(bitinfo, bitinfo_szi_True)


def test_get_bitinformation_label(rasm):
    """Test bp.get_bitinformation serializes when label given."""
    ds = rasm
    bp.get_bitinformation(ds, dim="x", label="rasm")
    assert os.path.exists("rasm.json")
    # second call should be faster
    bp.get_bitinformation(ds, dim="x", label="rasm")
    os.remove("rasm.json")


@pytest.mark.parametrize("dtype", ["Float64", "Float32", "Float16"])
def test_get_bitinformation_dtype(rasm, dtype):
    """Test bp.get_bitinformation returns correct number of bits depending on dtype."""
    ds = rasm.astype(dtype)
    v = list(ds.data_vars)[0]
    assert len(bp.get_bitinformation(ds, dim="x", label="rasm")[v]) == int(
        dtype.replace("Float", "")
    )
