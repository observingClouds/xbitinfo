import argparse
import json
import os

import numpy as np
import xarray as xr
from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Main  # noqa: E402

path_to_julia_functions = os.path.join(
    os.path.dirname(__file__), "get_n_plot_bitinformation.jl"
)
Main.path = path_to_julia_functions
jl.eval(
    'import Pkg; Pkg.add(["BitInformation", "NetCDF", "PyPlot", "StatsBase", "ColorSchemes"])'
)
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


def get_bitinformation(ds, dim=None, mask=None, label=None, overwrite=False):
    """Wrapper around BitInformation.jl

    Returns
    -------
    info_per_bit : dict
      Information content per bit and variable
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
            # nbits = ds[var].dtype.itemsize * 8
            X = ds[var].values
            Main.X = X
            info_per_bit[var] = jl.eval("get_bitinformation(X)")
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
    """Get the amount of bits to keep for a given information content

    Returns
    -------
    keepbits : dict
      Number of bits to keep per variable
    """

    def get_inflevel(var, inflevel):
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
    keepbits = get_keepbits(info_per_bit)
    print(keepbits)
