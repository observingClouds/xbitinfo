"""Tests for `xbitinfo.get_keepbits`."""
import pytest
import xarray as xr

import xbitinfo as xb


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
