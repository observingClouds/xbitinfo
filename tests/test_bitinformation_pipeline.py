import os

import pytest
import xarray as xr
from prefect.executors import DaskExecutor, LocalExecutor

import bitinformation_pipeline as bp


def test_full():
    """Test bitinformation_pipeline end to end."""
    label = "air_temperature"
    ds = xr.tutorial.load_dataset(label)
    # bitinformation_pipeline
    bitinfo = bp.get_bitinformation(ds, dim="lon")
    keepbits = bp.get_keepbits(bitinfo)
    # ds_bitrounded = bp.jl_bitround(ds, keepbits)
    ds_bitrounded = bp.xr_bitround(ds, keepbits)  # identical
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


@pytest.mark.parametrize("executor", [LocalExecutor, DaskExecutor])
def test_get_prefect_flow_executor(rasm, executor):
    """Test get_prefect_flow runs for different executors."""
    paths = []
    im = 3
    for i in range(im):
        f = f"file_{i}.nc"
        paths.append(f)
        rasm.to_netcdf(f)
    flow = bp.get_prefect_flow(paths)
    flow.run(executor)
    for i in range(im):
        assert os.path.exists(paths[i].replace(".nc", "_bitrounded_compressed.nc"))
        os.remove(paths[i])
        os.remove(paths[i].replace(".nc", "_bitrounded_compressed.nc"))
