import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from julia.api import Julia

jl = Julia(compiled_modules=False, debug=False)
from julia import Main  # noqa: E402

path_to_julia_functions = os.path.join(
    os.path.dirname(__file__), "get_n_plot_bitinformation.jl"
)
Main.path = path_to_julia_functions
jl.using("BitInformation")
jl.eval("include(Main.path)")


def get_user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="filename of dataset (netCDF-file) whose information "
        "content should be retrieved",
        type=str,
    )
    args = parser.parse_args()
    return args


def get_bitinformation(ds, label=None, overwrite=False, **kwargs):
    """Wrap BitInformation.bitinformation().

    Inputs
    ------
    ds : xr.Dataset
      input netcdf to analyse
    label : str
      label of the json to serialize bitinfo
    overwrite : bool
      if true, use serialized bitinfo based on label; if false, rerun bitinformation
    kwargs
      to be passed to bitinformation:
      - masked_value: defaults to NaN (different to bitinformation.jl)
      - mask: use masked_value instead
      - set_zero_insignificant (bool): defaults to True
      - confidence (float): defaults to 0.99

    Returns
    -------
    info_per_bit : dict
      Information content per bit and variable

    Example
    -------
        >>> ds = xr.tutorial.load_dataset("rasm")
        >>> bp.get_bitinformation(ds, dim="x")
        {'Tair': array([6.28759085e-01, 7.37993809e-01, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           4.10064704e-06, 4.75985410e-01, 5.20519662e-01, 3.93991763e-01,
           3.63939048e-01, 2.00036924e-01, 1.31092761e-01, 8.93712601e-02,
           7.16473010e-02, 6.84045398e-02, 6.87513712e-02, 6.89925122e-02,
           6.90843796e-02, 6.90237233e-02, 6.95050692e-02, 6.94720711e-02,
           6.94601453e-02, 6.96058765e-02, 6.98843472e-02, 6.92556384e-02,
           6.84707129e-02, 6.91220148e-02, 6.93995066e-02, 6.92542336e-02,
           6.88630993e-02, 6.88312736e-02, 6.89656830e-02, 6.93957020e-02,
           6.85981736e-02, 6.97206990e-02, 6.96303301e-02, 6.89981939e-02,
           7.03003113e-02, 6.96626582e-02, 6.94376911e-02, 6.91778910e-02,
           6.93997653e-02, 7.01042669e-02, 6.96544993e-02, 6.92199298e-02,
           6.97360327e-02, 6.95376714e-02, 6.97447985e-02, 6.95418140e-02,
           6.96346655e-02, 6.97496057e-02, 6.95058114e-02, 6.93239423e-02,
           6.89041586e-02, 6.95802295e-02, 6.96424276e-02, 6.56236800e-02,
           6.95009315e-02, 7.67737099e-02, 8.06336563e-02, 8.08729657e-02])}

    """
    if label is not None and overwrite is False:
        info_per_bit = load_bitinformation(label)
        if info_per_bit is None:
            overwrite = True
    if label is None:
        fn = ds.encoding["source"]
        label = fn
        overwrite = True
    if overwrite:
        info_per_bit = {}
        for var in ds.data_vars:
            X = ds[var].values
            Main.X = X
            if "mask" in kwargs:
                raise ValueError(
                    "bitinformation_pipeline does not wrap the mask argument. Mask your xr.Dataset with NaNs instead."
                )
            if "dim" in kwargs:
                if isinstance(kwargs["dim"], str):
                    kwargs["dim"] = ds[var].get_axis_num(kwargs["dim"]) + 1
            if "masked_value" not in kwargs:
                kwargs[
                    "masked_value"
                ] = f"convert({str(ds[var].dtype).capitalize()},NaN)"
            kwargs_str = ", " + ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            kwargs_str = kwargs_str.replace("True", "true").replace("False", "false")
            logging.debug(f"get_bitinformation(X{kwargs_str})")
            info_per_bit[var] = jl.eval(f"get_bitinformation(X{kwargs_str})")
        with open(label + ".json", "w") as f:
            json.dump(info_per_bit, f, cls=JsonCustomEncoder)
    return info_per_bit


def load_bitinformation(label):
    """Load bitinformation from JSON file"""
    label_file = label + ".json"
    if os.path.exists(label_file):
        with open(label_file) as f:
            info_per_bit = json.open(f)
        print(info_per_bit)
        return info_per_bit
    else:
        return None


def get_keepbits(ds, info_per_bit, inflevel=0.99):
    """Get the amount of bits to keep for a given information content.

    Inputs
    ------
    ds : xr.Dataset
      Dataset for which the information content has been retrieved
    info_per_bit : dict
      Information content of each bit for each variable in ds. This is the output from get_bitinformation.
    inflevel : float or dict
      Level of information that shall be preserved. Of type `float` if the
      preserved information content should be equal across variables, otherwise of type `dict`.

    Returns
    -------
    keepbits : dict
      Number of bits to keep per variable

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("rasm")
    >>> info_per_bit = bp.get_bitinformation(ds, dim="x")
    >>> bp.get_keepbits(ds, info_per_bit)
    {'Tair': 15}
    >>> bp.get_keepbits(ds, info_per_bit, inflevel=0.99999999)
    {'Tair': 15}
    """

    def get_inflevel(var, inflevel):
        """Helper function to load inflevel depending on input type."""
        if isinstance(inflevel, dict):
            return inflevel[var]
        else:
            return inflevel

    keepbits = {}
    config = {}
    for var in ds.data_vars:
        config[var] = {
            "inflevel": get_inflevel(var, inflevel),
            "bitinfo": info_per_bit[var],
            "maskinfo": int(ds[var].notnull().sum()),
        }
        Main.config = config[var]
        keepbits[var] = jl.eval("get_keepbits(config)")
    return keepbits


def plot_bitinformation(ds, bitinfo):
    """Plot bitwise information content
    Inputs
    ------
    bitinfo : dict
      Dictionary containing the bitwise information content for each variable
    Returns
    -------
    fig : matplotlib figure
    """

    import cmcrameri.cm as cmc

    nvars = len(bitinfo)
    varnames = sorted(bitinfo.keys())

    infbits_dict = get_keepbits(ds, bitinfo, 0.99)
    infbits100_dict = get_keepbits(ds, bitinfo, 0.999999999)

    ICnan = np.zeros((nvars, 64))
    infbits = infbits100 = np.zeros(nvars)
    ICnan[:, :] = np.nan
    for v, var in enumerate(varnames):
        ic = bitinfo[var]
        ICnan[v, : len(ic)] = ic
        infbits[v] = infbits_dict[var]
        infbits100[v] = infbits100_dict[var]
    ICnan = np.where(ICnan == 0, np.nan, ICnan)
    ICcsum = np.nancumsum(ICnan, axis=1)

    infbitsy = np.hstack([0, np.repeat(np.arange(1, nvars), 2), nvars])
    infbitsx = np.repeat(infbits, 2)
    infbitsx100 = np.repeat(infbits100, 2)

    fig_height = np.max([4, 4 + (nvars - 10) * 0.2])  # auto adjust to nvars
    fig, ax1 = plt.subplots(1, 1, figsize=(12, fig_height), sharey=True)
    ax1.invert_yaxis()
    ax1.set_box_aspect(1 / 32 * nvars)
    plt.tight_layout(rect=[0.06, 0.18, 0.8, 0.98])
    pos = ax1.get_position()
    cax = fig.add_axes([pos.x0, 0.12, pos.x1 - pos.x0, 0.02])

    ax1right = ax1.twinx()
    ax1right.invert_yaxis()
    ax1right.set_box_aspect(1 / 32 * nvars)

    cmap = cmc.turku_r
    pcm = ax1.pcolormesh(ICnan, vmin=0, vmax=1, cmap=cmap)
    cbar = plt.colorbar(pcm, cax=cax, orientation="horizontal")
    cbar.set_label("information content [bit]")

    # 99% of real information enclosed
    ax1.plot(
        np.hstack([infbits, infbits[-1]]),
        np.arange(nvars + 1),
        "C1",
        ds="steps-pre",
        zorder=10,
        label="99% of\ninformation",
    )

    # grey shading
    ax1.fill_betweenx(
        infbitsy, infbitsx, np.ones(len(infbitsx)) * 32, alpha=0.4, color="grey"
    )
    ax1.fill_betweenx(
        infbitsy, infbitsx100, np.ones(len(infbitsx)) * 32, alpha=0.1, color="c"
    )
    ax1.fill_betweenx(
        infbitsy,
        infbitsx100,
        np.ones(len(infbitsx)) * 32,
        alpha=0.3,
        facecolor="none",
        edgecolor="c",
    )

    # for legend only
    ax1.fill_betweenx(
        [-1, -1],
        [-1, -1],
        [-1, -1],
        color="burlywood",
        label="last 1% of\ninformation",
        alpha=0.5,
    )
    ax1.fill_betweenx(
        [-1, -1],
        [-1, -1],
        [-1, -1],
        facecolor="teal",
        edgecolor="c",
        label="false information",
        alpha=0.3,
    )
    ax1.fill_betweenx([-1, -1], [-1, -1], [-1, -1], color="w", label="unused bits")

    ax1.axvline(1, color="k", lw=1, zorder=3)
    ax1.axvline(9, color="k", lw=1, zorder=3)

    fig.suptitle(
        "Real bitwise information content",
        x=0.05,
        y=0.98,
        fontweight="bold",
        horizontalalignment="left",
    )

    ax1.set_xlim(0, 32)
    ax1.set_ylim(nvars, 0)
    ax1right.set_ylim(nvars, 0)

    ax1.set_yticks(np.arange(nvars) + 0.5)
    ax1right.set_yticks(np.arange(nvars) + 0.5)
    ax1.set_yticklabels(varnames)
    ax1right.set_yticklabels([f"{i:4.1f}" for i in ICcsum[:, -1]])
    ax1right.set_ylabel("total information per value [bit]")

    ax1.text(
        infbits[0] + 0.1,
        0.8,
        f"{infbits[0]-9} mantissa bits",
        fontsize=8,
        color="saddlebrown",
    )
    for i in range(1, nvars):
        ax1.text(
            infbits[i] + 0.1,
            (i) + 0.8,
            f"{infbits[i]-9}",
            fontsize=8,
            color="saddlebrown",
        )

    ax1.set_xticks([1, 9])
    ax1.set_xticks(np.hstack([np.arange(1, 8), np.arange(9, 32)]), minor=True)
    ax1.set_xticklabels([])
    ax1.text(0, nvars + 1.2, "sign", rotation=90)
    ax1.text(2, nvars + 1.2, "exponent bits", color="darkslategrey")
    ax1.text(10, nvars + 1.2, "mantissa bits")

    for i in range(1, 9):
        ax1.text(
            i + 0.5, nvars + 0.5, i, ha="center", fontsize=7, color="darkslategrey"
        )

    for i in range(1, 24):
        ax1.text(8 + i + 0.5, nvars + 0.5, i, ha="center", fontsize=7)

    ax1.legend(bbox_to_anchor=(1.08, 0.5), loc="center left", framealpha=0.6)

    fig.show()

    return fig


class JsonCustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (complex, np.complex)):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    args = get_user_input()
    ds = xr.open_mfdataset(args.filename)
    info_per_bit = get_bitinformation(ds)
    print(info_per_bit)
    keepbits = get_keepbits(ds, info_per_bit)
    print(keepbits)
