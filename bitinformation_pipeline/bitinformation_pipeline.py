import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from julia.api import Julia
from tqdm.auto import tqdm

jl = Julia(compiled_modules=False, debug=False)
from julia import Main  # noqa: E402

path_to_julia_functions = os.path.join(
    os.path.dirname(__file__), "get_n_plot_bitinformation.jl"
)
Main.path = path_to_julia_functions
jl.using("BitInformation")
jl.eval("include(Main.path)")


NMBITS = {64: 12, 32: 9, 16: 6}  # number of non mantissa bits for given dtype


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


def get_bitinformation(ds, dim=None, axis=None, label=None, overwrite=False, **kwargs):
    """Wrap BitInformation.bitinformation().

    Inputs
    ------
    ds : xr.Dataset
      input netcdf to analyse
    dim : str
      Dimension over which to apply mean. Only one of the `dim` and `axis` arguments can be supplied.
    axis : int
      Axis over which to apply mean. Only one of the `dim` and `axis` arguments can be supplied.
    label : str
      label of the json to serialize bitinfo
    overwrite : bool
      if false, try using serialized bitinfo based on label; if true or label does not exist, run bitinformation
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
        >>> ds = xr.tutorial.load_dataset("air_temperature")
        >>> bp.get_bitinformation(ds, dim="lon")
        {'air': array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00, 3.94447851e-01, 3.94447851e-01, 3.94447851e-01,
               3.94447851e-01, 3.94447851e-01, 3.94310542e-01, 7.36739987e-01,
               5.62682836e-01, 3.60511555e-01, 1.52471111e-01, 4.18818055e-02,
               3.65276146e-03, 1.19975820e-05, 4.39366160e-05, 4.18329296e-05,
               2.54572089e-05, 1.44121797e-04, 1.34144798e-03, 1.55468479e-06,
               5.38601212e-04, 8.09862581e-04, 1.74893445e-04, 4.97915410e-05,
               3.88027711e-04, 0.00000000e+00, 3.95323228e-05, 6.88854435e-04])}
    """
    if overwrite:
        calc = True
    else:
        calc = False
        if label is None:
            calc = True
        else:
            info_per_bit = load_bitinformation(label)
            if info_per_bit is None:
                calc = True
    if calc:
        # check keywords
        if (axis is None and dim is None) or (axis is not None and dim is not None):
            raise ValueError(
                "Please provide either `axis` or `dim` but not both or none."
            )
        if axis:
            if not isinstance(axis, int):
                raise ValueError(f"Please provide `axis` as `int`, found {type(axis)}.")
        if dim:
            if not isinstance(dim, str):
                raise ValueError(f"Please provide `dim` as `str`, found {type(dim)}.")
        if "mask" in kwargs:
            raise ValueError(
                "`bitinformation_pipeline` does not wrap the mask argument. Mask your xr.Dataset with NaNs instead."
            )

        info_per_bit = {}
        pbar = tqdm(ds.data_vars)
        for char in pbar:
            pbar.set_description("Processing %s" % char)
            X = ds[var].values
            Main.X = X
            if axis is not None:
                # in julia convention axis + 1
                dim = axis + 1
            if isinstance(dim, str):
                # in julia convention axis + 1
                dim = ds[var].get_axis_num(dim) + 1
            assert isinstance(dim, int)
            Main.dim = dim
            if "masked_value" not in kwargs:
                kwargs[
                    "masked_value"
                ] = f"convert({str(ds[var].dtype).capitalize()},NaN)"
            if "set_zero_insignificant" not in kwargs:
                kwargs["set_zero_insignificant"] = True
            kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            # convert python to julia bool
            kwargs_str = kwargs_str.replace("True", "true").replace("False", "false")
            logging.debug(f"get_bitinformation(X, dim={dim}, {kwargs_str})")
            info_per_bit[var] = jl.eval(
                f"get_bitinformation(X, dim={dim}, {kwargs_str})"
            )
        if label is not None:
            with open(label + ".json", "w") as f:
                logging.debug(f"Save bitinformation to {label+'.json'}")
                json.dump(info_per_bit, f, cls=JsonCustomEncoder)
    return info_per_bit


def load_bitinformation(label):
    """Load bitinformation from JSON file"""
    label_file = label + ".json"
    if os.path.exists(label_file):
        with open(label_file) as f:
            logging.debug(f"Load bitinformation from {label+'.json'}")
            info_per_bit = json.load(f)
        print(info_per_bit)
        return info_per_bit
    else:
        return None


def get_keepbits(info_per_bit, inflevel=0.99):
    """Get the number of mantissa bits to keep. To be used in xr_bitround and jl_bitround.

    Inputs
    ------
    info_per_bit : dict
      Information content of each bit for each variable in ds. This is the output from get_bitinformation.
    inflevel : float or dict
      Level of information that shall be preserved. Of type `float` if the
      preserved information content should be equal across variables, otherwise of type `dict`.

    Returns
    -------
    keepbits : dict
      Number of mantissa bits to keep per variable

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = bp.get_bitinformation(ds, dim="lon")
    >>> bp.get_keepbits(info_per_bit)
    {'air': 7}
    >>> bp.get_keepbits(info_per_bit, inflevel=0.99999999)
    {'air': 14}
    """
    keepmantissabits = {}
    for v, ic in info_per_bit.items():
        # set below threshold to zero
        # use something a bit bigger than maximum of the last 4 bits
        threshold = 1.5 * np.max(ic[-4:])
        ic_over_threshold = np.where(ic < threshold, 0, ic)
        ic_over_threshold_cum = np.cumsum(ic_over_threshold)  # CDF
        # normed CDF
        ic_over_threshold_cum_normed = ic_over_threshold_cum / ic_over_threshold_cum[-1]
        # return mantissabits to keep therefore subtract sign and exponent bits
        il = inflevel[v] if isinstance(inflevel, dict) else inflevel
        keepmantissabits[v] = (
            np.argmax(ic_over_threshold_cum_normed > il) + 1 - NMBITS[len(ic)]
        )
    return keepmantissabits


def _get_keepbits(ds, info_per_bit, inflevel=0.99):
    """Get the amount of mantissa bits to keep for a given information content.

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
      Number of mantissa bits to keep per variable

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = bp.get_bitinformation(ds, dim="lon")
    >>> bp._get_keepbits(ds, info_per_bit)
    {'air': 7}
    >>> bp._get_keepbits(ds, info_per_bit, inflevel=0.99999999)
    {'air': 14}
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
        # keep mantissa bits
        keepbits[var] = keepbits[var] - NMBITS[len(info_per_bit[var])]
    return keepbits


def _jl_bitround(X, keepbits):
    """Wrap BitInformation.round. Used in bp.jl_bitround."""
    Main.X = X
    Main.keepbits = keepbits
    return jl.eval("round!(X, keepbits)")


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
    info_per_bit = get_bitinformation(ds, axis=0)
    print(info_per_bit)
    keepbits = get_keepbits(info_per_bit)
    print(keepbits)
