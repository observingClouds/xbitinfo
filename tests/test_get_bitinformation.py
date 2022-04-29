#!/usr/bin/env python

"""Tests for `xbitinfo` package."""
import os

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_equal
from xarray.core import formatting
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.variable import Variable
from xarray.testing import assert_identical

import xbitinfo as xb


def assert_different(a, b):
    """Raises an AssertionError if two objects are equal. This will match
    data values, dimensions and coordinates, but not names or attributes
    (except for Dataset objects for which the variable names must match).
    Arrays with NaN in the same location are considered equal.
    Parameters
    ----------
    a : xarray.Dataset, xarray.DataArray or xarray.Variable
        The first object to compare.
    b : xarray.Dataset, xarray.DataArray or xarray.Variable
        The second object to compare.
    See Also
    --------
    assert_identical, assert_allclose, Dataset.equals, DataArray.equals
    numpy.testing.assert_array_equal
    """
    __tracebackhide__ = True
    assert type(a) == type(b)
    if isinstance(a, (Variable, DataArray)):
        assert not a.equals(b), formatting.diff_array_repr(a, b, "equals")
    elif isinstance(a, Dataset):
        assert not a.equals(b), formatting.diff_dataset_repr(a, b, "equals")
    else:
        raise TypeError(f"{type(a)} not supported by assertion comparison")


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


def test_get_bitinformation_returns_dataaset():
    """Test xb.get_bitinformation returns dict."""
    ds = xr.tutorial.load_dataset("rasm")
    assert isinstance(xb.get_bitinformation(ds, axis=0), xr.Dataset)


def test_get_bitinformation_dim():
    """Test xb.get_bitinformation is sensitive to dim."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo0 = xb.get_bitinformation(ds, axis=0)
    bitinfo2 = xb.get_bitinformation(ds, axis=2)
    assert_different(bitinfo0, bitinfo2)


def test_get_bitinformation_dim_string_equals_axis_int():
    """Test xb.get_bitinformation undestands xarray dimension names the same way as axis as integers."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfox = xb.get_bitinformation(ds, dim="x")
    bitinfo2 = xb.get_bitinformation(ds, axis=2)
    assert_identical(bitinfox, bitinfo2)


def test_get_bitinformation_masked_value():
    """Test xb.get_bitinformation is sensitive to masked_value."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo = xb.get_bitinformation(ds, dim="x")
    bitinfo_no_mask = xb.get_bitinformation(ds, dim="x", masked_value="nothing")
    bitinfo_no_mask_None = xb.get_bitinformation(ds, dim="x", masked_value=None)
    assert_identical(bitinfo_no_mask, bitinfo_no_mask_None)
    assert_different(bitinfo, bitinfo_no_mask)


def test_get_bitinformation_set_zero_insignificant():
    """Test xb.get_bitinformation is sensitive to set_zero_insignificant."""
    ds = xr.tutorial.load_dataset("air_temperature")
    dim = "lon"
    bitinfo_szi_False = xb.get_bitinformation(ds, dim=dim, set_zero_insignificant=False)
    bitinfo_szi_True = xb.get_bitinformation(ds, dim=dim, set_zero_insignificant=True)
    bitinfo = xb.get_bitinformation(ds, dim=dim)
    assert_different(bitinfo, bitinfo_szi_False)
    assert_identical(bitinfo, bitinfo_szi_True)


def test_get_bitinformation_confidence():
    """Test xb.get_bitinformation is sensitive to confidence."""
    ds = xr.tutorial.load_dataset("air_temperature")
    dim = "lon"
    bitinfo_conf99 = xb.get_bitinformation(ds, dim=dim, confidence=0.99)
    bitinfo_conf50 = xb.get_bitinformation(ds, dim=dim, confidence=0.5)
    bitinfo = xb.get_bitinformation(ds, dim=dim)
    assert_different(bitinfo_conf99, bitinfo_conf50)
    assert_identical(bitinfo, bitinfo_conf99)


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
