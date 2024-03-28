import matplotlib.pyplot as plt
import pytest
import xarray as xr

import xbitinfo as xb
from xbitinfo.graphics import add_bitinfo_labels, plot_bitinformation


def test_add_bitinfo_labels():
    ds = xr.tutorial.load_dataset("air_temperature")
    info_per_bit = xb.get_bitinformation(ds, dim="lon")
    inflevels = [1.0, 0.9999, 0.99, 0.975, 0.95]
    keepbits = [23, 14, 7, 6, 5]
    ds_bitrounded_along_lon = xb.bitround.bitround_along_dim(
        ds, info_per_bit, dim="lon", inflevels=inflevels
    )
    diff = (ds - ds_bitrounded_along_lon)["air"].isel(time=0)
    ax = plt.gca()
    diff.plot()

    with pytest.raises(KeyError):
        add_bitinfo_labels(diff, info_per_bit, inflevels, keepbits)

    add_bitinfo_labels(diff, info_per_bit, inflevels)

    # Check if a Matplotlib figure object is created
    assert plt.gcf() is not None

    # Check if the text labels were added to the plot
    assert len(ax.texts) == 2 * len(inflevels)

    # Check if the plot contains the expected number of lines
    assert len(ax.lines) == len(inflevels)

    # Check if the labels have the correct content
    if inflevels is None:
        expected_inflevels = ["100.0%", "100.0%", "99.88%", "98.89%", "95.28%"]
        for i, keep in enumerate(keepbits):
            inf_text = expected_inflevels[i]
            keepbits_text = f"keepbits = {keep}"
            assert ax.texts[i].get_text() == inf_text
            assert ax.texts[i + 5].get_text() == keepbits_text

    if keepbits is None:
        expected_keepbits = keepbits
        for i, inf in enumerate(inflevels):
            inf_text = str(round(inf * 100, 2)) + "%"
            keepbits_text = expected_keepbits[i]
            assert ax.texts[i].get_text() == inf_text
            assert ax.texts[i + 5].get_text() == keepbits_text
    # Cleanup the plot
    plt.close()


@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_plot_bitinformation(dtype):
    rasm = xr.tutorial.load_dataset("air_temperature")
    ds = rasm.astype(dtype)
    info_per_bit = xb.get_bitinformation(ds, dim="lon")
    plot_bitinformation(info_per_bit)
