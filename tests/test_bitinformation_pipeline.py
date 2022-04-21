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


im = 3


@pytest.fixture()
def flow_paths(rasm):
    paths = []
    for i in range(im):
        f = f"file_{i}.nc"
        paths.append(f)
        rasm.to_netcdf(f)
    flow = bp.get_prefect_flow(paths)
    return flow, paths


@pytest.mark.parametrize("executor", [LocalExecutor, DaskExecutor])
def test_get_prefect_flow_executor(flow_paths, executor):
    """Test get_prefect_flow runs for different executors."""
    flow, paths = flow_paths
    flow.run(executor)


def test_get_prefect_flow_inflevel_parameter(flow, flow_paths):
    """Test get_prefect_flow runs for different parameters."""
    flow, paths = flow_paths
    flow.run(parameters=dict(inflevel=0.90))
    os.move("file_0_bitrounded_compressed.nc", "file_0_bitrounded_compressed_bu.nc")

    flow.run(parameters=dict(inflevel=0.99999999))

    inflevel090 = xr.open_dataset("file_0_bitrounded_compressed_bu.nc")
    inflevel099999999 = xr.open_dataset("file_1_bitrounded_compressed.nc")
    assert not inflevel090.equals(inflevel099999999)
    assert (
        inflevel090.Tair.attrs["_QuantizeBitRoundNumberOfSignificantDigits"]
        <= inflevel099999999.Tair.attrs["_QuantizeBitRoundNumberOfSignificantDigits"]
    )


def test_cleanup(flow_paths):
    flow, paths = flow_paths
    # cleanup
    for i in range(im):
        os.remove(paths[i])
        os.remove(paths[i].replace(".nc", "_bitrounded_compressed.nc"))
