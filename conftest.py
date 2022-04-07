import numpy as np
import pytest
import xarray as xr
from xarray.tutorial import load_dataset

import bitinformation_pipeline as bp

xr.set_options(display_style="text")


@pytest.fixture(autouse=True)
def add_standard_imports(
    doctest_namespace,
):
    """imports for doctest"""
    xr.set_options(display_style="text")
    doctest_namespace["np"] = np
    doctest_namespace["xr"] = xr
    doctest_namespace["bp"] = bp
    # always seed numpy.random to make the examples deterministic
    np.random.seed(42)


@pytest.fixture()
def rasm():
    return load_dataset("rasm").astype("float32")  # dtype conversion is cheating here


@pytest.fixture()
def air_temperature():
    return load_dataset("air_temperature")


@pytest.fixture()
def ROMS_example():
    return load_dataset("ROMS_example")


@pytest.fixture()
def era52mt():
    return load_dataset("era5-2mt-2019-03-uk.grib")


@pytest.fixture()
def eraint_uvz():
    return load_dataset("eraint_uvz")
