"""Tests for `xbitinfo.get_keepbits`."""
import pytest
import xarray as xr

import xbitinfo as xb
from xbitinfo.xbitinfo import get_keepbits


@pytest.fixture
def rasm_info_per_bit(rasm):
    return xb.get_bitinformation(rasm, axis=0)


def test_get_keepbits(rasm_info_per_bit):
    """Test xb.get_keepbits returns xr.Dataset."""
    assert isinstance(xb.get_keepbits(rasm_info_per_bit), xr.Dataset)


def test_get_keepbits_inflevel_1(rasm_info_per_bit):
    """Test xb.get_keepbits returns all mantissa bits for inflevel 1."""
    keepbits = xb.get_keepbits(rasm_info_per_bit, inflevel=1)
    assert (keepbits == 53).all()


@pytest.mark.parametrize("inflevel", [0.99, [0.99, 1.0]])
def test_get_keepbits_inflevel_dim(rasm_info_per_bit, inflevel):
    """Test xb.get_keepbits returns inflevel as dim."""
    keepbits = xb.get_keepbits(rasm_info_per_bit, inflevel=inflevel)
    assert "inflevel" in keepbits.dims
    if isinstance(inflevel, (int, float)):
        inflevel = [inflevel]
    assert (keepbits.inflevel == inflevel).all()


def test_get_keepbits_informationFilter():
    ds = xr.tutorial.load_dataset("air_temperature")
    info = xb.get_bitinformation(ds, dim="lat")
    var = info["air"]
    for i in range(var.size):
        if i >= 19 and i <= 24:
            var[i] = 0.05
    keepbits_dataset = get_keepbits(
        info,
        inflevel=[0.90],
        information_filter="On",
        **{"threshold": 0.7, "tolerance": 0.001}
    )
    keepbits = keepbits_dataset["air"].values
    assert keepbits == 5


def test_get_keepbits_informationFilter_1():
    ds = xr.tutorial.load_dataset("air_temperature")
    info = xb.get_bitinformation(ds, dim="lat")
    keepbitsOff_dataset = get_keepbits(
        info,
        inflevel=[0.99],
        information_filter="Off",
        **{"threshold": 0.7, "tolerance": 0.001}
    )
    keepbits_Off = keepbitsOff_dataset["air"].values

    keepbitsOn_dataset = get_keepbits(
        info,
        inflevel=[0.99],
        information_filter="On",
        **{"threshold": 0.7, "tolerance": 0.001}
    )
    keepbits_On = keepbitsOn_dataset["air"].values

    assert keepbits_Off == keepbits_On
