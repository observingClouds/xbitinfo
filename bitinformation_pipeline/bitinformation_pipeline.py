import argparse
import json
import os

import numpy as np
from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Main  # noqa: E402

path_to_julia_functions = os.path.join(
    os.path.dirname(__file__), "get_n_plot_bitinformation.jl"
)
Main.path = path_to_julia_functions
jl.eval('import Pkg; Pkg.add("BitInformation"); Pkg.add("NetCDF");')
jl.eval('Pkg.add("PyPlot"), Pkg.add("StatsBase"); Pkg.add("ColorSchemes");')
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


def get_bitinformation(filename, dim=None, mask=None, label=None, overwrite=False):
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
        fn = os.path.basename(filename)
        label = fn
        overwrite = True
    if overwrite:
        Main.inputfile = filename
        info_per_bit = jl.eval("get_bitinformation(inputfile)")
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


def get_keepbits(info_per_bit, information_content=0.99):
    """Get the amount of bits to keep for a given information content

    Returns
    -------
    keepbits : dict
      Number of bits to keep per variable
    """
    Main.info_per_bit = info_per_bit
    Main.information_content = information_content
    keepbits = jl.eval("get_keepbits(info_per_bit, information_content)")
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
    info_per_bit = get_bitinformation(args.filename)
    print(info_per_bit)
    keepbits = get_keepbits(info_per_bit)
    print(keepbits)
