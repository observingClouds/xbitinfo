import pytest
import xarray as xr
from dask import is_dask_collection
from xarray.testing import assert_allclose, assert_equal

import bitinformation_pipeline as bp


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("implementation", ["xarray", "julia"])
@pytest.mark.parametrize("input_type", ["Dataset", "DataArray"])
@pytest.mark.parametrize("keepbits", ["dict", "int"])
def test_xr_bitround(air_temperature, input_type, implementation, keepbits, inplace):
    """Test xr_bitround to different keepbits of type dict or int."""
    ds = air_temperature
    i = 15
    if keepbits == "dict":
        keepbits = {v: i for v in ds.data_vars}
    elif keepbits == "int":
        keepbits = i
    if input_type == "DataArray":
        v = list(ds.data_vars)[0]
        ds = ds[v]

    bitround = bp.xr_bitround if implementation == "xarray" else bp.jl_bitround
    ds_bitrounded = bitround(ds, keepbits, inplace=inplace)

    def check(da, da_bitrounded, inplace):
        if inplace:
            assert_equal(da, da_bitrounded)
        else:
            # close but
            assert_allclose(da, da_bitrounded, atol=0.01, rtol=0.01)
            # different after bitrounding
            diff = (da - da_bitrounded).compute()
            assert (diff != 0).any()

        # attrs set
        assert da_bitrounded.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] == i

    if input_type == "DataArray":
        check(ds, ds_bitrounded, inplace)
    else:
        for v in ds.data_vars:
            check(ds[v], ds_bitrounded[v], inplace)


@pytest.mark.parametrize(
    "dask",
    [pytest.param(True, marks=pytest.mark.skip(reason="require .values")), False],
)
@pytest.mark.parametrize("implementation", ["xarray", "julia"])
def test_xr_bitround_dask(air_temperature, implementation, dask):
    """Test xr_bitround keeps dask."""
    ds = air_temperature
    i = 15
    keepbits = i
    if dask:
        ds = ds.chunk("auto")

    bitround = bp.xr_bitround if implementation == "xarray" else bp.jl_bitround
    ds_bitrounded = bitround(ds, keepbits)
    assert is_dask_collection(ds_bitrounded) == dask


@pytest.mark.parametrize("keepbits", list(range(1, 23)))
def test_bitround_xarray_julia_equal(air_temperature, keepbits):
    """Test jl_bitround and xr_bitround yield identical results."""
    ds = air_temperature
    ds_xr_bitrounded = bp.xr_bitround(ds, keepbits)
    ds_jl_bitrounded = bp.jl_bitround(ds, keepbits)
    assert_equal(ds_jl_bitrounded, ds_xr_bitrounded)
