import argparse

from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Main  # noqa: E402

jl.eval('import Pkg; Pkg.add("BitInformation"); Pkg.add("NetCDF");')
jl.eval('Pkg.add("PyPlot"), Pkg.add("StatsBase"); Pkg.add("ColorSchemes");')
jl.using("BitInformation")
jl.eval('include("get_n_plot_bitinformation.jl")')


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


def get_bitinformation(filename, dim=None, mask=None, label=None):
    """Wrapper around BitInformation.jl

    Returns
    -------
    info_per_bit : dict
      Information content per bit and variable
    """
    Main.inputfile = filename
    info_per_bit = jl.eval("get_bitinformation(inputfile)")
    return info_per_bit


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


if __name__ == "__main__":
    args = get_user_input()
    info_per_bit = get_bitinformation(args.filename)
    print(info_per_bit)
    keepbits = get_keepbits(info_per_bit)
    print(keepbits)
