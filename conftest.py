import numpy as np
import pooch
import pytest
import xarray as xr
from xarray.tutorial import load_dataset

import xbitinfo as xb

xr.set_options(display_style="text")


@pytest.fixture(autouse=True)
def add_standard_imports(
    doctest_namespace,
):
    """imports for doctest"""
    xr.set_options(display_style="text")
    doctest_namespace["np"] = np
    doctest_namespace["xr"] = xr
    doctest_namespace["xb"] = xb
    # always seed numpy.random to make the examples deterministic
    np.random.seed(42)


@pytest.fixture()
def rasm():
    """one atmospheric variable float64 with masked ocean"""
    return load_dataset("rasm")


@pytest.fixture()
def air_temperature():
    """one atmospheric variable float32 over the US no mask"""
    return load_dataset("air_temperature")


@pytest.fixture()
def ROMS_example():
    """two ocean variabls float32 with masked land"""
    return load_dataset("ROMS_example")


@pytest.fixture()
def era52mt():
    """one variable float32 t2m over the UK no mask"""
    return load_dataset("era5-2mt-2019-03-uk.grib")


@pytest.fixture()
def eraint_uvz():
    """three atmospheric variable float32 global no mask"""
    return load_dataset("eraint_uvz")


@pytest.fixture()
def ugrid_demo():
    """sea surface height of a Tsunami simulation with ICON"""
    return xr.open_dataset(pooch.retrieve(url="https://psyplot.github.io/examples/_downloads/3fe9a9cde72c892e7e26accd0a57cff8/ugrid_demo.nc", known_hash="80d75f8f3a68cc254aa7f725a8f8eab10a1f794a4576453f97395f95f928ad83"))


@pytest.fixture()
def icon_grid_demo():
    """Temperature, zonal and meridional wind simulated with ICON"""
    return xr.open_dataset(pooch.retrieve(url="https://psyplot.github.io/examples/_downloads/c8ccf6e61c8a76db0065720e09d2ed6e/icon_grid_demo.nc", known_hash="9725c6264122a5018d488b81959ddb4708698b1e2314a73b595f90561617d9e5"))
