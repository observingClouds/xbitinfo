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
    assert isinstance(a, type(b))
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


def bitinfo_assert_allclose(bitinfo1, bitinfo2, **kwargs):
    assert list(bitinfo1.keys()) == list(bitinfo2.keys()), print(
        f"lhs = {bitinfo1.keys()} vs rhs = {bitinfo2.keys()}"
    )
    for v in bitinfo1.keys():
        assert_allclose(bitinfo1[v], bitinfo2[v], **kwargs)


def bitinfo_assert_different(bitinfo1, bitinfo2):
    """Fail bitinfo different values."""
    assert (bitinfo1 != bitinfo2).any()


@pytest.mark.parametrize("implementation", ["julia", "python"])
def test_get_bitinformation_returns_dataset(implementation):
    """Test xb.get_bitinformation returns xr.Dataset."""
    ds = xr.tutorial.load_dataset("rasm")
    assert isinstance(
        xb.get_bitinformation(ds, implementation=implementation, axis=0), xr.Dataset
    )


@pytest.mark.parametrize("implementation", ["julia", "python"])
def test_get_bitinformation_dim(implementation):
    """Test xb.get_bitinformation is sensitive to dim."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo0 = xb.get_bitinformation(ds, axis=0, implementation=implementation)
    bitinfo2 = xb.get_bitinformation(ds, axis=2, implementation=implementation)
    assert_different(bitinfo0, bitinfo2)


@pytest.mark.parametrize("implementation", ["julia", "python"])
def test_get_bitinformation_dim_string_equals_axis_int(implementation):
    """Test xb.get_bitinformation undestands xarray dimension names the same way as axis as integers."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfox = xb.get_bitinformation(ds, dim="x", implementation=implementation)
    bitinfo2 = xb.get_bitinformation(ds, axis=2, implementation=implementation)
    assert_identical(bitinfox, bitinfo2)


def test_get_bitinformation_masked_value(implementation="julia"):
    """Test xb.get_bitinformation is sensitive to masked_value."""
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo = xb.get_bitinformation(ds, dim="x", implementation=implementation)
    bitinfo_no_mask = xb.get_bitinformation(
        ds, dim="x", masked_value="nothing", implementation=implementation
    )
    bitinfo_no_mask_None = xb.get_bitinformation(
        ds, dim="x", masked_value=None, implementation=implementation
    )
    assert_identical(bitinfo_no_mask, bitinfo_no_mask_None)
    assert_different(bitinfo, bitinfo_no_mask)


@pytest.mark.parametrize("implementation", ["julia", "python"])
def test_get_bitinformation_set_zero_insignificant(implementation):
    """Test xb.get_bitinformation is sensitive to set_zero_insignificant."""
    ds = xr.tutorial.load_dataset("air_temperature")
    dim = "lon"
    bitinfo = xb.get_bitinformation(ds, dim=dim, implementation=implementation)
    bitinfo_szi_False = xb.get_bitinformation(
        ds, dim=dim, set_zero_insignificant=False, implementation=implementation
    )
    try:
        bitinfo_szi_True = xb.get_bitinformation(
            ds, dim=dim, set_zero_insignificant=True, implementation=implementation
        )
        assert_identical(bitinfo, bitinfo_szi_True)
    except NotImplementedError:
        assert implementation == "python"
    if implementation == "python":
        assert_identical(bitinfo, bitinfo_szi_False)
    elif implementation == "julia":
        assert_different(bitinfo, bitinfo_szi_False)


@pytest.mark.parametrize("implementation", ["julia", "python"])
def test_get_bitinformation_confidence(implementation):
    """Test xb.get_bitinformation is sensitive to confidence."""
    ds = xr.tutorial.load_dataset("air_temperature")
    dim = "lon"
    bitinfo = xb.get_bitinformation(ds, dim=dim, implementation=implementation)
    try:
        bitinfo_conf99 = xb.get_bitinformation(
            ds, dim=dim, confidence=0.99, implementation=implementation
        )
        bitinfo_conf50 = xb.get_bitinformation(
            ds, dim=dim, confidence=0.5, implementation=implementation
        )
        assert_different(bitinfo_conf99, bitinfo_conf50)
        assert_identical(bitinfo, bitinfo_conf99)
    except AssertionError:
        assert implementation == "python"


@pytest.mark.parametrize("implementation", ["julia", "python"])
def test_get_bitinformation_label(rasm, implementation):
    """Test xb.get_bitinformation serializes when label given."""
    ds = rasm
    xb.get_bitinformation(
        ds, dim="x", label="./tmp_testdir/rasm", implementation=implementation
    )
    assert os.path.exists("./tmp_testdir/rasm.json")
    # second call should be faster
    xb.get_bitinformation(
        ds, dim="x", label="./tmp_testdir/rasm", implementation=implementation
    )
    os.remove("./tmp_testdir/rasm.json")


@pytest.mark.parametrize("implementation", ["julia", "python"])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_get_bitinformation_dtype(rasm, dtype, implementation):
    """Test xb.get_bitinformation returns correct number of bits depending on dtype."""
    ds = rasm.astype(dtype)
    v = list(ds.data_vars)[0]
    dtype_bits = dtype.replace("float", "")
    assert len(xb.get_bitinformation(ds, dim="x")[v].coords["bit" + dtype]) == int(
        dtype_bits
    )


@pytest.mark.parametrize("implementation", ["julia", "python"])
def test_get_bitinformation_multidim(rasm, implementation):
    """Test xb.get_bitinformation runs on all dimensions by default"""
    ds = rasm
    bi = xb.get_bitinformation(ds, implementation=implementation)
    # check length of dimension
    assert bi.dims["dim"] == len(ds.dims)
    bi_time = bi.sel(dim="time").Tair.values
    bi_x = bi.sel(dim="x").Tair.values
    bi_y = bi.sel(dim="y").Tair.values
    assert any(bi_time != bi_x)
    assert any(bi_time != bi_y)
    assert any(bi_y != bi_x)


@pytest.mark.parametrize("implementation", ["julia", "python"])
def test_get_bitinformation_different_variables_dims(rasm, implementation):
    """Test xb.get_bitinformation runs with variables of different dimensionality"""
    ds = rasm
    # add variable with different dimensionality
    ds["Tair_mean"] = ds.Tair.mean(dim="time")
    bi = xb.get_bitinformation(ds, implementation=implementation)
    assert all(np.isnan(bi.Tair_mean.sel(dim="time")))
    bi_Tair_mean_x = bi.Tair_mean.sel(dim="x")
    bi_Tair_x = bi.Tair.sel(dim="x")
    assert_different(bi_Tair_mean_x, bi_Tair_x)


@pytest.mark.parametrize("implementation", ["julia", "python"])
def test_get_bitinformation_different_dtypes(rasm, implementation):
    ds = rasm
    ds["Tair32"] = ds.Tair.astype("float32")
    ds["Tair16"] = ds.Tair.astype("float16")
    bi = xb.get_bitinformation(ds, implementation=implementation)
    for bitdim in ["bitfloat16", "bitfloat32", "bitfloat64"]:
        assert bitdim in bi.dims
        assert bitdim in bi.coords


@pytest.mark.parametrize("implementation", ["julia", "python"])
def test_get_bitinformation_dim_list(rasm, implementation):
    bi = xb.get_bitinformation(rasm, dim=["x", "y"], implementation=implementation)
    assert (bi.dim == ["x", "y"]).all()


def test_get_bitinformation_keep_attrs(rasm):
    bi = xb.get_bitinformation(rasm, dim=["x", "y"]).Tair
    assert "units" in bi.attrs
    assert bi.attrs["units"] == 1
    for a in rasm.Tair.attrs.keys():
        assert bi.attrs["source_" + a] == rasm.Tair.attrs[a], print(bi.attrs)


@pytest.mark.parametrize(
    "ds,dim,axis",
    [
        ("ugrid_demo", None, -1),
        ("icon_grid_demo", "ncells", None),
        ("air_temperature", "lon", None),
        ("rasm", "x", None),
        ("ROMS_example", "eta_rho", None),
        ("era52mt", "time", None),
        ("eraint_uvz", "longitude", None),
    ],
)
def test_implementations_agree(ds, dim, axis, request):
    """Test whether the python and julia implementation retrieve the same results"""
    ds = request.getfixturevalue(ds)
    bi_python = xb.get_bitinformation(
        ds,
        dim=dim,
        axis=axis,
        implementation="python",
        set_zero_insignificant=False,
        overwrite=True,
        masked_value=None,
    )
    bi_julia = xb.get_bitinformation(
        ds,
        dim=dim,
        axis=axis,
        implementation="julia",
        set_zero_insignificant=False,
        overwrite=True,
        masked_value=None,
    )
    bitinfo_assert_allclose(bi_python, bi_julia, rtol=1e-4)
