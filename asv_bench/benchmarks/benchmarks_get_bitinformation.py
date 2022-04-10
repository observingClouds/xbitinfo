import warnings

import numpy as np
import xarray as xr

from bitinformation_pipeline import get_bitinformation

from . import _skip_slow, ensure_loaded, parameterized, randn, requires_dask

warnings.filterwarnings("ignore", message="Index.ravel returning ndarray is deprecated")


class Base:
    """
    Benchmark time and peak memory of `get_bitinformation`.
    """

    # https://asv.readthedocs.io/en/stable/benchmarks.html
    timeout = 300.0
    repeat = 1
    number = 5

    def setup(self, *args, **kwargs):
        raise NotImplementedError()

    def time_get_bitinformation(self, **kwargs):
        """Take time for `get_bitinformation`."""
        self.info_per_bit = ensure_loaded(
            get_bitinformation(self.ds, dim=self.dim, **kwargs)
        )

    def peakmem_get_bitinformation(self, **kwargs):
        """Take memory peak for `get_bitinformation`."""
        self.info_per_bit = ensure_loaded(
            get_bitinformation(self.ds, dim=self.dim, **kwargs)
        )


class xr_tutorial_datasets(Base):
    def setup(self, *args, **kwargs):
        raise NotImplementedError()

    def get_data(self, label="rasm", dim="x"):
        self.ds = xr.tutorial.load_dataset(label)
        self.dim = dim


class rasm(xr_tutorial_datasets):
    def setup(self, *args, **kwargs):
        self.get_data(label="rasm", dim="x", **kwargs)


class air_temperature(xr_tutorial_datasets):
    def setup(self, *args, **kwargs):
        self.get_data(label="air_temperature", dim="lon", **kwargs)


class Random(Base):
    """
    Generate random input data.
    """

    def get_data(self, dim="x", spatial_res=5, ntime=120):
        """Generates random number xr.Dataset."""
        self.dim = dim
        self.time = ntime
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
                    (self.ntime, self.nx, self.ny),
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
                    "history": "created for bitinformation_pipeline benchmarking",
                },
            )
            .squeeze()
            .to_dataset()
        )

    def setup(self, *args, **kwargs):
        self.get_data()
