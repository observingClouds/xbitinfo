import pytest
import xarray as xr
from xarray.testing import assert_allclose

import bitinformation_pipeline as bp


@pytest.mark.parametrize("input_type", ["Dataset", "DataArray"])
@pytest.mark.parametrize("keepbits", ["dict", "int"])
def test_xr_bitround(air_temperature, input_type, keepbits):
    """Test xr_bitround to different keepbits of type dict or int."""
    ds = air_temperature
    i = 15
    if keepbits == "dict":
        keepbits = {v: i for v in ds.data_vars}
    elif keepbits == "int":
        keepbits = i
    if input_type == "DataArray":
        ds = ds.to_array()

    ds_bitrounded = bp.xr_bitround(ds, keepbits)

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
