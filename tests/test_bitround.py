import pytest
import xarray as xr
from dask import is_dask_collection
from xarray.testing import assert_allclose, assert_equal

import xbitinfo as xb


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("implementation", ["xarray", "julia"])
@pytest.mark.parametrize("input_type", ["Dataset", "DataArray"])
@pytest.mark.parametrize("keepbits", ["dict", "int"])
def test_xr_bitround(air_temperature, dtype, input_type, implementation, keepbits):
    """Test xr_bitround to different keepbits of type dict or int."""
    ds = air_temperature.astype(dtype)
    i = 6
    if keepbits == "dict":
        keepbits = {v: i for v in ds.data_vars}
    elif keepbits == "int":
        keepbits = i
    if input_type == "DataArray":
        v = list(ds.data_vars)[0]
        ds = ds[v]

    bitround = xb.xr_bitround if implementation == "xarray" else xb.jl_bitround
    ds_bitrounded = bitround(ds, keepbits)

    def check(da, da_bitrounded):
        # check close
        assert_allclose(da, da_bitrounded, atol=0.01, rtol=0.01)
        # attrs set
        assert da_bitrounded.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] == i
        # different after bitrounding
        diff = (da - da_bitrounded).compute()
        assert (diff != 0).any()

    if input_type == "DataArray":
        check(ds, ds_bitrounded)
    else:
        for v in ds.data_vars:
            check(ds[v], ds_bitrounded[v])


@pytest.mark.parametrize(
    "implementation,dask",
    [("xarray", True), ("xarray", False), ("julia", False)],
)
def test_bitround_dask(air_temperature, implementation, dask):
    """Test xr_bitround and jl_bitround keeps dask and successfully computes."""
    ds = air_temperature
    i = 15
    keepbits = i
    if dask:
        ds = ds.chunk("auto")

    bitround = xb.xr_bitround if implementation == "xarray" else xb.jl_bitround
    ds_bitrounded = bitround(ds, keepbits)
    assert is_dask_collection(ds_bitrounded) == dask
    if dask:
        assert ds_bitrounded.compute()


@pytest.mark.parametrize(
    "dtype,keepbits",
    [("float16", range(1, 9)), ("float32", range(1, 23)), ("float64", range(1, 52))],
)
def test_bitround_xarray_julia_equal(air_temperature, dtype, keepbits):
    """Test jl_bitround and xr_bitround yield identical results."""
    ds = air_temperature.astype(dtype)
    for keep in keepbits:
        ds_xr_bitrounded = xb.xr_bitround(ds, keep)
        ds_jl_bitrounded = xb.jl_bitround(ds, keep)
        assert_equal(ds_jl_bitrounded, ds_xr_bitrounded)
