import os
import shutil

import numcodecs
import pytest
import xarray as xr
import zarr

import xbitinfo as xb


@pytest.mark.parametrize("for_cdo", [True, False])
def test_get_compress_encoding_for_cdo(rasm, for_cdo):
    ds = rasm
    encoding = xb.get_compress_encoding_nc(ds, for_cdo=for_cdo)
    v = list(ds.data_vars)[0]
    time_axis = ds[v].get_axis_num("time")
    if for_cdo:
        assert encoding[v]["chunksizes"][time_axis] == 1
    else:
        assert encoding[v]["chunksizes"][time_axis] > 1


@pytest.mark.parametrize("dask", [True, False])
def test_to_compressed_netcdf(rasm, dask):
    """Test to_compressed_netcdf reduces size on disk."""
    ds = rasm
    if dask:
        ds = ds.chunk("auto")
    label = "file"
    # save
    ds.to_netcdf(f"./tmp_testdir/{label}.nc")
    ds.to_compressed_netcdf(f"./tmp_testdir/{label}_compressed.nc")
    # check size reduction
    ori_size = os.path.getsize(f"./tmp_testdir/{label}.nc")
    compressed_size = os.path.getsize(f"./tmp_testdir/{label}_compressed.nc")
    assert compressed_size < ori_size


def test_to_compressed_netcdf_for_cdo_no_time_dim_var(air_temperature):
    """Test to_compressed_netcdf if `for_cdo=True` and one var without `time_dim`."""
    ds = air_temperature
    ds["air_mean"] = ds["air"].isel(time=0)
    ds.to_compressed_netcdf("./tmp_testdir/test.nc", for_cdo=True)
    os.remove("./tmp_testdir/test.nc")


def get_zarr_size(fn):
    """Get size of zarr file excluding metadata"""
    # Open file
    grp = zarr.open_group(fn)
    # Collect size
    total = 0
    for var in list(grp.keys()):
        total += grp[var].nbytes_stored
    return total


def test_to_compressed_zarr(rasm):
    """Test to_compressed_zarr reduces size on disk."""
    ds = rasm
    label = "file"
    # save
    encoding = {
        var: {"compressor": None} for var in ds.data_vars
    }  # deactivate default compression
    ds.to_zarr(f"./tmp_testdir/{label}.zarr", mode="w", encoding=encoding)
    ds.to_compressed_zarr(f"./tmp_testdir/{label}_compressed.zarr", mode="w")
    # check size reduction
    ori_size = get_zarr_size(f"./tmp_testdir/{label}.zarr")
    compressed_size = get_zarr_size(f"./tmp_testdir/{label}_compressed.zarr")
    assert compressed_size < ori_size
    shutil.rmtree(f"./tmp_testdir/{label}.zarr")
    shutil.rmtree(f"./tmp_testdir/{label}_compressed.zarr")


def test_to_compressed_zarr_individual_compressors(eraint_uvz):
    """Test non-default compressors."""
    ds = eraint_uvz
    label = "file"
    # save
    encoding = {
        var: {"compressor": None} for var in ds.data_vars
    }  # deactivate default compression
    ds.to_zarr(f"./tmp_testdir/{label}.zarr", mode="w", encoding=encoding)
    compressors = {
        "u": numcodecs.Blosc("zstd", clevel=7),
        "v": numcodecs.Blosc("zlib", clevel=7, shuffle=numcodecs.Blosc.BITSHUFFLE),
    }
    ds.to_compressed_zarr(
        f"./tmp_testdir/{label}_compressed.zarr", compressors, mode="w"
    )
    # check size reduction
    ori_size = get_zarr_size(f"./tmp_testdir/{label}.zarr")
    compressed_size = get_zarr_size(f"./tmp_testdir/{label}_compressed.zarr")
    assert compressed_size < ori_size
    shutil.rmtree(f"./tmp_testdir/{label}.zarr")
    shutil.rmtree(f"./tmp_testdir/{label}_compressed.zarr")
