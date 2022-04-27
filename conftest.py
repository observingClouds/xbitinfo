import numpy as np
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
