import os

import prefect
import pytest
import xarray as xr
from prefect_dask.task_runners import DaskTaskRunner

import xbitinfo as xb

if __name__ == "__main__":

    @pytest.mark.parametrize(
        "ds,dim,axis",
        [
            ("ugrid_demo", None, -1),
            ("icon_grid_demo", "ncells", None),
            ("air_temperature", "lon", None),
            ("rasm", "x", None),
            ("ROMS_example", "eta_rho", None),
            ("era52mt", "time", None),
            ("eraint_uvz", "longitude", None),
        ],
    )
    def test_full(ds, dim, axis, request):
        """Test xbitinfo end to end."""
        ds = request.getfixturevalue(ds)
        # xbitinfo
        bitinfo = xb.get_bitinformation(ds, dim=dim, axis=axis)
        keepbits = xb.get_keepbits(bitinfo)
        # ds_bitrounded = xb.jl_bitround(ds, keepbits)
        ds_bitrounded = xb.xr_bitround(ds, keepbits)  # identical
        # save
        label = os.path.basename(ds.encoding["source"])
        print(label)
        ds.to_netcdf(f"./tmp_testdir/{label}.nc")
        ds.to_compressed_netcdf(f"./tmp_testdir/{label}_compressed.nc")
        ds_bitrounded.to_compressed_netcdf(
            f"./tmp_testdir/{label}_bitrounded_compressed.nc"
        )
        # check size reduction
        ori_size = os.path.getsize(f"./tmp_testdir/{label}.nc")
        compressed_size = os.path.getsize(f"./tmp_testdir/{label}_compressed.nc")
        bitrounded_compressed_size = os.path.getsize(
            f"./tmp_testdir/{label}_bitrounded_compressed.nc"
        )
        if dim == "eta_rho":
            assert (
                compressed_size < ori_size * 1.1
            )  # previous compression is already really good for ROMS_example
        else:
            assert compressed_size < ori_size
        assert bitrounded_compressed_size < compressed_size

    def test_full_max_keepbits():
        """Test pipeline to get maximum keepbits"""
        label = "air_temperature"
        ds = xr.tutorial.load_dataset(label)
        bi = xb.get_bitinformation(ds)
        kb = xb.get_keepbits(bi)
        kb_max = kb.max(dim="dim")
        _ = xb.plot_bitinformation(bi.isel(dim=[0]))
        _ = xb.plot_bitinformation(bi.isel(dim=0))
        ds_bitrounded_max = xb.xr_bitround(ds, kb_max)
        ds_bitrounded_max.to_compressed_zarr(f"{label}.zarr", mode="w")

    imax = 3

    @pytest.fixture()
    def flow_paths(rasm):
        paths = []
        stride = rasm.time.size // imax
        for i in range(imax):
            f = f"./tmp_testdir/file_{i}.nc"
            if os.path.exists(f.replace(".nc", "_bitrounded_compressed.nc")):
                os.remove(f.replace(".nc", "_bitrounded_compressed.nc"))
            paths.append(f)
            rasm.isel(time=slice(stride * i, stride * (i + 1) - 1)).to_netcdf(f)
        flow = xb.get_prefect_flow(paths)
        yield flow, paths
        # cleanup
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
            if os.path.exists(p.replace(".nc", "_bitrounded_compressed.nc")):
                os.remove(p.replace(".nc", "_bitrounded_compressed.nc"))

    @pytest.mark.parametrize(
        "executor",
        [
            "local",
            "my_dask_client",
            pytest.param(
                "DaskExecutor",
                marks=pytest.mark.skip(reason="fails with few resources"),
            ),
            pytest.param(
                "LocalDaskExecutor",
                marks=pytest.mark.skip(reason="fails with few resources"),
            ),
        ],
    )
    def test_get_prefect_flow_executor(flow_paths, executor):
        """Test get_prefect_flow runs for different executors."""
        flow, paths = flow_paths
        for f in paths:
            if os.path.exists(f.replace(".nc", "_bitrounded_compressed.nc")):
                os.remove(f.replace(".nc", "_bitrounded_compressed.nc"))

        if executor == "local":
            flow.run()
        elif executor == "my_dask_client":
            from dask.distributed import Client

            client = Client(n_workers=4, threads_per_worker=1, processes=True)
            # point Prefect's DaskTaskRunner to our Dask cluster
            dask_executor = DaskTaskRunner(address=client.scheduler.address)
            flow.run(executor=dask_executor)
            client.close()
        else:
            # For other executors, you can use the Prefect executor directly
            executor_class = getattr(prefect.executors, executor)
            flow.run(executor=executor_class())

    def test_get_prefect_flow_inflevel_parameter(flow_paths):
        """Test get_prefect_flow runs for different parameters."""
        flow, paths = flow_paths
        st090 = flow.run(parameters={"axis": -1, "inflevel": 0.90, "overwrite": True})
        st099999999 = flow.run(
            parameters={"axis": -1, "inflevel": 0.99999999, "overwrite": True}
        )
        keepbits = flow.get_tasks(name="get_bitinformation_keepbits")[0]

        assert st099999999.result[keepbits].result != st090.result[keepbits].result
