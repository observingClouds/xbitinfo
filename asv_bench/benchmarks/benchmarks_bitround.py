import numpy as np
import xarray as xr

from xbitinfo import jl_bitround, xr_bitround

from . import (
    _skip_julia,
    _skip_slow,
    ensure_loaded,
    parameterized,
    randn,
    requires_dask,
    requires_distributed,
)


class Base:
    """
    Benchmark time and peak memory of `xr_bitround` and `jl_bitround`.
    """

    # https://asv.readthedocs.io/en/stable/benchmarks.html
    timeout = 300.0
    repeat = 1
    number = 5

    def setup(self, *args, **kwargs):
        raise NotImplementedError()

    def time_xr_bitround(self, **kwargs):
        """Take time for `xr_bitround`."""
        ensure_loaded(xr_bitround(self.ds, self.keepbits, **kwargs))

    def peakmem_xr_bitround(self, **kwargs):
        """Take memory peak for `xr_bitround`."""
        ensure_loaded(xr_bitround(self.ds, self.keepbits, **kwargs))

    def time_jl_bitround(self, **kwargs):
        """Take time for `jl_bitround`."""
        ensure_loaded(jl_bitround(self.ds, self.keepbits, **kwargs))

    def peakmem_jl_bitround(self, **kwargs):
        """Take memory peak for `jl_bitround`."""
        ensure_loaded(jl_bitround(self.ds, self.keepbits, **kwargs))

    peakmem_jl_bitround.setup = _skip_julia
    time_jl_bitround.setup = _skip_julia


class xr_tutorial_datasets(Base):
    def setup(self, *args, **kwargs):
        raise NotImplementedError()

    def get_data(self, label="rasm", keepbits=7):
        self.ds = xr.tutorial.load_dataset(label)
        self.keepbits = keepbits


class rasm(xr_tutorial_datasets):
    def setup(self, *args, **kwargs):
        self.get_data(label="rasm", keepbits=7, **kwargs)


class air_temperature(xr_tutorial_datasets):
    def setup(self, *args, **kwargs):
        self.get_data(label="air_temperature", keepbits=7, **kwargs)


class Random(Base):
    """
    Generate random input data.
    """

    def get_data(self, keepbits=7, spatial_res=5, ntime=120, dtype="float32"):
        """Generates random number xr.Dataset."""
        self.keepbits = keepbits
        self.ntime = ntime
        self.nx = 360 // spatial_res
        self.ny = 360 // spatial_res

        FRAC_NAN = 0.0

        times = xr.cftime_range(start="2000", freq="MS", periods=ntime)
        lons = xr.DataArray(
            np.linspace(0.5, 359.5, self.nx),
            dims=("lon",),
            attrs={"units": "degrees east", "long_name": "longitude"},
        )
        lats = xr.DataArray(
            np.linspace(-89.5, 89.5, self.ny),
            dims=("lat",),
            attrs={"units": "degrees north", "long_name": "latitude"},
        )
        self.ds = (
            xr.DataArray(
                randn(
                    (self.nx, self.ny, self.ntime),
                    frac_nan=FRAC_NAN,
                ),
                coords={
                    "lon": lons,
                    "lat": lats,
                    "time": times,
                },
                dims=("lon", "lat", "time"),
                name="var",
                attrs={
                    "units": "var units",
                    "description": "a description",
                    "history": "created for xbitinfo benchmarking",
                },
            )
            .squeeze()
            .to_dataset()
            .astype(dtype)
        )

    def setup(self, *args, **kwargs):
        self.get_data(keepbits=7)


class RandomDask(Random):
    def setup(self, *args, **kwargs):
        requires_dask()
        self.get_data(keepbits=7, spatial_res=1, ntime=12 * 100)  # 100yr monthly 1deg
        self.ds = self.ds.chunk("auto")


class RandomDaskClient(RandomDask):
    def setup(self, *args, **kwargs):
        requires_distributed()
        requires_dask()
        _skip_slow()
        from dask.distributed import Client

        self.client = Client()
        super().setup(**kwargs)

    def cleanup(self):
        self.client.shutdown()
