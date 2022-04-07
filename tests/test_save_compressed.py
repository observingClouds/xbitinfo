import os

import pytest
import xarray as xr

import bitinformation_pipeline as bp


@pytest.mark.parametrize("for_cdo", [True, False])
def test_get_compress_encoding_for_cdo(rasm, for_cdo):
    ds = rasm
    encoding = bp.get_compress_encoding(ds, for_cdo=for_cdo)
    v = list(ds.data_vars)[0]
    time_axis = ds[v].get_axis_num("time")
    if for_cdo:
        assert encoding[v]["chunksizes"][time_axis] == 1
    else:
        assert encoding[v]["chunksizes"][time_axis] > 1


def test_to_compressed_netcdf(rasm):
    """Test bitinformation_pipeline end to end."""
    ds = rasm
    label = "file"
    # save
    ds.to_netcdf(f"{label}.nc")
    ds.to_compressed_netcdf(f"{label}_compressed.nc")
    # check size reduction
    ori_size = os.path.getsize(f"{label}.nc")
    compressed_size = os.path.getsize(f"{label}_compressed.nc")
    assert compressed_size < ori_size
