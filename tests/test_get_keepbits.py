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


def test_get_keepbits_informationFilter():
    """
    Test the `get_keepbits` function with different information filters.

    This test function checks the behavior of the `get_keepbits` function when applying gradient information filter.
    The dataset contains artificial information and thus applying the filter should result in lesser number of bits
    than what should be when filter is None.


    Raises:
        AssertionError: If the test conditions are not met.

    """

    bit32_values = [
        "±",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "e8",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "m7",
        "m8",
        "m9",
        "m10",
        "m11",
        "m12",
        "m13",
        "m14",
        "m15",
        "m16",
        "m17",
        "m18",
        "m19",
        "m20",
        "m21",
        "m22",
        "m23",
    ]
    data_variable = xr.DataArray(
        data=[
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.11799129e-01,
            8.19114977e-01,
            4.41578500e-01,
            3.25470303e-01,
            4.35195738e-01,
            2.81462993e-01,
            2.10719742e-01,
            1.46638224e-01,
            9.24534031e-02,
            4.41090879e-02,
            1.13842504e-02,
            8.20088050e-04,
            2.62239097e-06,
            7.11284508e-07,
            1.18183485e-06,
            9.49338973e-09,
            1.80859255e-07,
            7.72662891e-07,
            1.37865391e-05,
            2.11117224e-06,
            2.01353088e-07,
            3.20755770e-02,
            6.06721012e-03,
            2.25987148e-04,
            1.71530452e-06,
            5.13067595e-03,
        ],
        coords={"bitfloat32": bit32_values, "dim": "x"},
        dims=["bitfloat32"],
    )
    info_ds = xr.Dataset({"RH2": data_variable})
    Keepbits_FilterNone = xb.get_keepbits(
        info_ds,
        inflevel=[0.99],
        information_filter=None,
        **{"threshold": 0.7, "tolerance": 0.001},
    )
    Keepbits_FilterNone_Value = Keepbits_FilterNone["RH2"].values
    assert Keepbits_FilterNone_Value == 19

    Keepbits_FilterGradient = xb.get_keepbits(
        info_ds,
        inflevel=[0.99],
        information_filter="Gradient",
        **{"threshold": 0.7, "tolerance": 0.001},
    )
    Keepbits_FilterGradient_Value = Keepbits_FilterGradient["RH2"].values
    assert Keepbits_FilterGradient_Value == 7


def test_get_keepbits_informationFilter_1():
    """
    Test the `get_keepbits` function with different information filters.

    This test function checks the behavior of the `get_keepbits` function when applying gradient information filter.
    The dataset does not contain artificial information and thus the number of keepbits when gradient filter is applied
    should be equal to when filter is None.

    Raises:
        AssertionError: If the test conditions are not met.

    """

    ds = xr.tutorial.load_dataset("air_temperature")
    info = xb.get_bitinformation(ds, dim="lat")
    Keepbits_FilterNone = xb.get_keepbits(
        info,
        inflevel=[0.99],
        information_filter=None,
        **{"threshold": 0.7, "tolerance": 0.001},
    )
    Keepbits_FilterNone_Value = Keepbits_FilterNone["air"].values

    Keepbits_FilterGradient = xb.get_keepbits(
        info,
        inflevel=[0.99],
        information_filter="Gradient",
        **{"threshold": 0.7, "tolerance": 0.001},
    )

    Keepbits_FilterGradient_Value = Keepbits_FilterGradient["air"].values
    assert Keepbits_FilterNone_Value == Keepbits_FilterGradient_Value
