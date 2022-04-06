import xarray as xr
from xarray.testing import assert_allclose

import bitinformation_pipeline as bp


def test_xr_bitround(rasm):
    """Test xr_bitround."""
    ds = rasm
    i = 15
    keepbits = {v: i for v in ds.data_vars}
    ds_bitrounded = bp.xr_bitround(ds, keepbits)
    assert_allclose(ds, ds_bitrounded)
    for v in ds.data_vars:
        # attrs set
        assert ds_bitrounded[v].attrs["bitround_keepbits"] == i
        # different after bitrounding
        assert ((ds[v] - ds_bitrounded[v]) != 0).all()
        # but close
        assert_allclose(ds, ds_bitrounded, atol=0.01, rtol=0.01)


def test_xr_bitround_keepbits_int(rasm):
    """Test xr_bitround with keepbits as int."""
    ds = rasm
    i = 15
    keepbits = i
    ds_bitrounded = bp.xr_bitround(ds, keepbits)
    assert_allclose(ds, ds_bitrounded)
    for v in ds.data_vars:
        # attrs set
        assert ds_bitrounded[v].attrs["bitround_keepbits"] == i
        # different after bitrounding
        assert ((ds[v] - ds_bitrounded[v]) != 0).all()
        # but close
        assert_allclose(ds, ds_bitrounded, atol=0.01, rtol=0.01)
