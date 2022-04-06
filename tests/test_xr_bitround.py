import xarray as xr
from xarray.testing import assert_allclose

import bitinformation_pipeline as bp


def test_xr_bitround():
    """Test xr_bitround."""
    label = "rasm"
    ds = xr.tutorial.load_dataset(label)
    i = 15
    keepbits = {v: i for v in ds.data_vars}
    ds_bitrounded = bp.xr_bitround(ds, keepbits)
    assert ds_bitrounded.attrs["bitround_keepbits"] == i
    assert_allclose(ds, ds_bitrounded)
    for v in ds.data_vars:
        assert ((ds[v] - ds_bitrounded[v]) != 0).all()
