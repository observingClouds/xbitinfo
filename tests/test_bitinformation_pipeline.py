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


imax = 3


@pytest.fixture()
def flow_paths(rasm):
    paths = []
    stride = rasm.time.size // imax
    for i in range(imax):
        f = f"file_{i}.nc"
        if os.path.exists(f.replace(".nc", "_bitrounded_compressed.nc")):
            os.remove(f.replace(".nc", "_bitrounded_compressed.nc"))
        paths.append(f)
        rasm.isel(time=slice(stride * i, stride * (i + 1) - 1)).to_netcdf(f)
    flow = bp.get_prefect_flow(paths)
    yield flow, paths
    # cleanup
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
        if os.path.exists(p.replace(".nc", "_bitrounded_compressed.nc")):
            os.remove(p.replace(".nc", "_bitrounded_compressed.nc"))


@pytest.mark.parametrize("executor", [LocalExecutor, DaskExecutor])
def test_get_prefect_flow_executor(flow_paths, executor):
    """Test get_prefect_flow runs for different executors."""
    flow, paths = flow_paths
    flow.run(executor=executor)


def test_get_prefect_flow_inflevel_parameter(flow_paths):
    """Test get_prefect_flow runs for different parameters."""
    flow, paths = flow_paths
    st090 = flow.run(parameters=dict(axis=-1, inflevel=0.90))
    st099999999 = flow.run(parameters=dict(axis=-1, inflevel=0.99999999))
    keepbits = flow.get_tasks(name="get_bitinformation_keepbits")[0]
    assert (
        st099999999.result[keepbits]._result.value
        != st090.result[keepbits]._result.value
    )
