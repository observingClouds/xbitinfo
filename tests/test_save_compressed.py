import os

import xarray as xr

import bitinformation_pipeline as bp


def test_to_compressed_netcdf():
    """Test bitinformation_pipeline end to end."""
    label = "rasm"
    ds = xr.tutorial.load_dataset(label)
    # save
    ds.to_netcdf(f"{label}.nc")
    ds.to_compressed_netcdf(f"{label}_compressed.nc")
    # check size reduction
    ori_size = os.path.getsize(f"{label}.nc")
    compressed_size = os.path.getsize(f"{label}_compressed.nc")
    assert compressed_size < ori_size
