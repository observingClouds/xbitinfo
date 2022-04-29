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
    os.path.dirname(__file__), "bitinformation_wrapper.jl"
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


def get_bit_coords(dtype_size):
    """Get coordinates for bits assuming float dtypes."""
    if dtype_size == 16:
        coords = (
            ["±"]
            + [f"e{int(i)}" for i in range(1, 6)]
            + [f"m{int(i-8)}" for i in range(6, 16)]
        )
    elif dtype_size == 32:
        coords = (
            ["±"]
            + [f"e{int(i)}" for i in range(1, 9)]
            + [f"m{int(i-8)}" for i in range(9, 32)]
        )
    elif dtype_size == 64:
        coords = (
            ["±"]
            + [f"e{int(i)}" for i in range(1, 12)]
            + [f"m{int(i-11)}" for i in range(12, 64)]
        )
    else:
        raise ValueError(f"dtype of size {dtype_size} neither known nor implemented.")
    return coords


def dict_to_dataset(info_per_bit):
    """Convert keepbits dictionary to dataset."""
    dsb = xr.Dataset()
    for v in info_per_bit.keys():
        dtype_size = len(info_per_bit[v])
        dim_name = f"bit{dtype_size}"
        dsb[v] = xr.DataArray(
            info_per_bit[v],
            dims=[dim_name],
            coords={dim_name: get_bit_coords(dtype_size)},
            name=v,
        ).astype("float16")
    return dsb


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
      - masked_value: defaults to `NaN` (different to bitinformation.jl), set `None` disable masking
      - mask: use `masked_value` instead
      - set_zero_insignificant (bool): defaults to `True`
      - confidence (float): defaults to 0.99

    Returns
    -------
    info_per_bit : dict
      Information content per bit and variable

    Example
    -------
        >>> ds = xr.tutorial.load_dataset("air_temperature")
        >>> xb.get_bitinformation(ds, dim="lon")
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
                "`xbitinfo` does not wrap the mask argument. Mask your xr.Dataset with NaNs instead."
            )

        info_per_bit = {}
        pbar = tqdm(ds.data_vars)
        for var in pbar:
            pbar.set_description("Processing %s" % var)
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
            elif kwargs["masked_value"] is None:
                kwargs["masked_value"] = "nothing"
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
    return dict_to_dataset(info_per_bit)


def load_bitinformation(label):
    """Load bitinformation from JSON file"""
    label_file = label + ".json"
    if os.path.exists(label_file):
        with open(label_file) as f:
            logging.debug(f"Load bitinformation from {label+'.json'}")
            info_per_bit = json.load(f)
        return dict_to_dataset(info_per_bit)
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
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> xb.get_keepbits(info_per_bit)
    {'air': 7}
    >>> xb.get_keepbits(info_per_bit, inflevel=0.99999999)
    {'air': 14}
    >>> xb.get_keepbits(info_per_bit, inflevel=1.0)
    {'air': 23}
    """
    keepmantissabits = {}
    if isinstance(inflevel, (int, float)):
        if inflevel < 0 or inflevel > 1.0:
            raise ValueError("Please provide `inflevel` from interval [0.,1.]")
    for v, ic in info_per_bit.items():
        if inflevel == 1.0:
            keepmantissabits[v] = len(ic) - NMBITS[len(ic)]
        else:
            # set below threshold to zero
            # use something a bit bigger than maximum of the last 4 bits
            threshold = 1.5 * np.max(ic[-4:])
            ic_over_threshold = np.where(ic < threshold, 0, ic)
            ic_over_threshold_cum = np.nancumsum(ic_over_threshold)  # CDF
            # normed CDF
            ic_over_threshold_cum_normed = (
                ic_over_threshold_cum / ic_over_threshold_cum[-1]
            )
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
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> xb._get_keepbits(ds, info_per_bit)
    {'air': 7}
    >>> xb._get_keepbits(ds, info_per_bit, inflevel=0.99999999)
    {'air': 14}
    >>> xb._get_keepbits(ds, info_per_bit, inflevel=1.0)
    {'air': -8}
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
    """Wrap BitInformation.round. Used in xb.jl_bitround."""
    Main.X = X
    Main.keepbits = keepbits
    return jl.eval("round!(X, keepbits)")


def get_prefect_flow(paths=[]):
    """
    Create prefect.Flow for xbitinfo bitrounding paths.

    1. Analyse bitwise real information content
    2. Retrieve keepbits
    3. Apply bitrounding with `xr_bitround`
    4. Save as compressed netcdf with `to_compressed_netcdf`

    Many parameters can be changed when running the flow `flow.run(parameters=dict(chunk="auto"))`:
    - paths: list of Paths
        Paths to be bitrounded
    - analyse_paths: str or int
        Which paths to be passed to `xb.get_bitinformation`. choose from ["first_last", "all", int], where int is interpreted as stride, i.e. paths[::stride]. Defaults to "first".
    - enforce_dtype : str or None
        Enforce dype for all variables. Currently `get_bitinformation` fails for different dtypes in variables. Do nothing if None. Defaults to None.
    - label : see get_bitinformation
    - dim/axis : see get_bitinformation
    - inflevel : see get_keepbits
    - non_negative_keepbits : bool
        Set negative keepbits from `get_keepbits` to 0. Required when using `xr_bitround`. Defaults to True.
    - chunks : see https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html. Note that with `chunks=None`, `dask` is not used for I/O and the flow is still parallelized when using `DaskExecutor`.
    - bitround_in_julia : bool
        Use `jl_bitround` instead of `xr_bitround`. Both should yield identical results. Defaults to False.
    - overwrite : bool
        Whether to overwrite bitrounded netcdf files. False (default) skips existing files.
    - complevel : see to_compressed_netcdf, defaults to 7.
    - rename : list
        Replace mapping for paths towards new_path of bitrounded file, i.e. replace=[".nc", "_bitrounded_compressed.nc"]

    Inputs
    ------
    paths : list
      list of Paths of files to be processed by `get_bitinformation`, `get_keepbits`, `xr_bitround` and `to_compressed_netcdf`.

    Returns
    -------
    prefect.Flow
      see https://docs.prefect.io/core/concepts/flows.html#overview

    Example
    -------
    Imagine n files of identical structure, i.e. 1-year per file climate model output:
    >>> ds = xr.tutorial.load_dataset("rasm")
    >>> year, datasets = zip(*ds.groupby("time.year"))
    >>> paths = [f"{y}.nc" for y in year]
    >>> xr.save_mfdataset(datasets, paths)

    Create prefect.Flow and run sequentially
    >>> flow = xb.get_prefect_flow(paths=paths)
    >>> import prefect
    >>> logger = prefect.context.get("logger")
    >>> logger.setLevel("ERROR")
    >>> st = flow.run()

    Inspect flow state
    >>> # flow.visualize(st)  # requires graphviz

    Run in parallel with dask:
    >>> import os  # https://docs.xarray.dev/en/stable/user-guide/dask.html
    >>> os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    >>> from prefect.executors import DaskExecutor, LocalDaskExecutor
    >>> from dask.distributed import Client
    >>> client = Client(n_workers=2, threads_per_worker=1, processes=True)
    >>> executor = DaskExecutor(
    ...     address=client.scheduler.address
    ... )  # take your own client
    >>> executor = DaskExecutor()  # use dask from prefect
    >>> executor = LocalDaskExecutor()  # use dask local from prefect
    >>> # flow.run(executor=executor, parameters=dict(overwrite=True))

    Modify parameters of a flow:
    >>> flow.run(parameters=dict(inflevel=0.9999, overwrite=True))
    <Success: "All reference tasks succeeded.">

    See also
    --------
    - https://examples.dask.org/applications/prefect-etl.html
    - https://docs.prefect.io/core/getting_started/basic-core-flow.html

    """

    from prefect import Flow, Parameter, task, unmapped
    from prefect.engine.signals import SKIP

    from .bitround import jl_bitround, xr_bitround

    @task
    def get_bitinformation_keepbits(
        paths,
        analyse_paths="first",
        label=None,
        inflevel=0.99,
        enforce_dtype=None,
        non_negative_keepbits=True,
        **get_bitinformation_kwargs,
    ):
        # take subset only for analysis in bitinformation
        if analyse_paths == "first_last":
            p = [paths[0], paths[-1]]
        elif analyse_paths == "all":
            p = paths
        elif analyse_paths == "first":
            p = paths[0]
        elif isinstance(analyse_paths, int):  # interpret as stride
            p = paths[::analyse_paths]
        else:
            raise ValueError(
                "Please provide analyse_paths as int interpreted as stride or from ['first_last','all','first','last']."
            )
        ds = xr.open_mfdataset(p)
        if enforce_dtype:
            ds = ds.astype(enforce_dtype)
        info_per_bit = get_bitinformation(ds, label=label, **get_bitinformation_kwargs)
        keepbits = get_keepbits(info_per_bit, inflevel=inflevel)
        if non_negative_keepbits:
            keepbits = {v: max(0, k) for v, k in keepbits.items()}  # ensure no negative
        return keepbits

    @task
    def bitround_and_save(
        path,
        keepbits,
        chunks=None,
        complevel=4,
        rename=[".nc", "_bitrounded_compressed.nc"],
        overwrite=False,
        enforce_dtype=None,
        bitround_in_julia=False,
    ):
        new_path = path.replace(rename[0], rename[1])
        if not overwrite:
            if os.path.exists(new_path):
                try:
                    ds_new = xr.open_dataset(new_path, chunks=chunks)
                    ds = xr.open_dataset(path, chunks=chunks)
                    if (
                        ds.nbytes == ds_new.nbytes
                    ):  # bitrounded and original have same number of bytes in memory
                        raise SKIP(f"{new_path} already exists.")
                except Exception as e:
                    print(
                        f"{type(e)} when xr.open_dataset({new_path}), therefore delete and recalculate."
                    )
                    os.remove(new_path)
        ds = xr.open_dataset(path, chunks=chunks)
        if enforce_dtype:
            ds = ds.astype(enforce_dtype)
        bitround_func = jl_bitround if bitround_in_julia else xr_bitround
        ds_bitround = bitround_func(ds, keepbits)
        ds_bitround.to_compressed_netcdf(new_path, complevel=complevel)
        return

    with Flow("xbitinfo") as flow:
        if paths == []:
            raise ValueError("Please provide paths of files to bitround, found [].")
        paths = Parameter("paths", default=paths)
        analyse_paths = Parameter("analyse_paths", default="first")
        dim = Parameter("dim", default=None)
        axis = Parameter("axis", default=0)
        inflevel = Parameter("inflevel", default=0.99)
        label = Parameter("label", default=None)
        rename = Parameter("rename", default=[".nc", "_bitrounded_compressed.nc"])
        overwrite = Parameter("overwrite", default=False)
        bitround_in_julia = Parameter("bitround_in_julia", default=False)
        complevel = Parameter("complevel", default=7)
        chunks = Parameter("chunks", default=None)
        enforce_dtype = Parameter("enforce_dtype", default=None)
        non_negative_keepbits = Parameter("non_negative_keepbits", default=True)
        keepbits = get_bitinformation_keepbits(
            paths,
            analyse_paths=analyse_paths,
            dim=dim,
            axis=axis,
            inflevel=inflevel,
            label=label,
            enforce_dtype=enforce_dtype,
            non_negative_keepbits=non_negative_keepbits,
        )  # once
        bitround_and_save.map(
            paths,
            keepbits=unmapped(keepbits),
            rename=unmapped(rename),
            chunks=unmapped(chunks),
            complevel=unmapped(complevel),
            overwrite=unmapped(overwrite),
            enforce_dtype=unmapped(enforce_dtype),
            bitround_in_julia=unmapped(bitround_in_julia),
        )  # parallel map
    return flow


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
