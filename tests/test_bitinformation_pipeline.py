import os

import xarray as xr

import bitinformation_pipeline as bp


def test_full():
    """Test bitinformation_pipeline end to end."""
    label = "air_temperature"
    ds = xr.tutorial.load_dataset(label)
    # bitinformation_pipeline
    bitinfo = bp.get_bitinformation(ds, dim="lon")
    keepbits = bp.get_keepbits(ds, bitinfo)
    ds_bitrounded = bp.jl_bitround(ds, keepbits)
    # save
    ds.to_netcdf(f"{label}.nc")
    ds.to_compressed_netcdf(f"{label}_compressed.nc")
    ds_bitrounded.to_compressed_netcdf(f"{label}_bitrounded_compressed.nc")
    # check size reduction
    ori_size = os.path.getsize(f"{label}.nc")
    compressed_size = os.path.getsize(f"{label}_compressed.nc")
    bitrounded_compressed_size = os.path.getsize(f"{label}_bitrounded_compressed.nc")
    assert compressed_size < ori_size
    assert bitrounded_compressed_size < compressed_size
