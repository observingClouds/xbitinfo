import os
import shutil

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


def pytest_sessionstart(session):
    """Run before start of tests"""
    os.makedirs("./tmp_testdir")


def pytest_sessionfinish(session):
    """Run after finishing tests"""
    shutil.rmtree("./tmp_testdir")


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
    return xr.open_dataset(
        pooch.retrieve(
            url="https://psyplot.github.io/examples/_downloads/3fe9a9cde72c892e7e26accd0a57cff8/ugrid_demo.nc",
            known_hash=None,
        )
    )[
        [
            "Mesh2_height",
            "Mesh2_bathy",
            "Mesh2_m_x",
            "Mesh2_m_y",
            "Mesh2_u_x",
            "Mesh2_u_y",
        ]
    ]


@pytest.fixture()
def icon_grid_demo():
    """Temperature, zonal and meridional wind simulated with ICON"""
    return xr.open_dataset(
        pooch.retrieve(
            url="https://psyplot.github.io/examples/_downloads/c8ccf6e61c8a76db0065720e09d2ed6e/icon_grid_demo.nc",
            known_hash=None,
        )
    )[["t2m", "u", "v"]]
