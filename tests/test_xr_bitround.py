import pytest
import xarray as xr
from xarray.testing import assert_allclose

import bitinformation_pipeline as bp


@pytest.mark.parametrize("keepbits", ["dict", "int"])
def test_xr_bitround(air_temperature, keepbits):
    """Test xr_bitround to different keepbits of type dict or int."""
    ds = air_temperature
    i = 15
    if keepbits == "dict":
        keepbits = {v: i for v in ds.data_vars}
    elif keepbits == "int":
        keepbits = i
    ds_bitrounded = bp.xr_bitround(ds, keepbits)
    assert_allclose(ds, ds_bitrounded, atol=0.01, rtol=0.01)
    for v in ds.data_vars:
        # attrs set
        assert ds_bitrounded[v].attrs["_QuantizeBitRoundNumberOfSignificantDigits"] == i
        # different after bitrounding
        diff = (ds[v] - ds_bitrounded[v]).compute()
        assert (diff != 0).any()
