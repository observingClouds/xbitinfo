import json
import logging
import os

import numpy as np
import xarray as xr
from dask import array as da

try:
    from julia.api import Julia

    julia_installed = True
except ImportError:
    julia_installed = False
from tqdm.auto import tqdm

from . import __version__
from . import _py_bitinfo as pb
from .julia_helpers import install

already_ran = False
if not already_ran and julia_installed:
    already_ran = install(quiet=True)
    jl = Julia(compiled_modules=False, debug=False)
    from julia import Main  # noqa: E402

    path_to_julia_functions = os.path.join(
        os.path.dirname(__file__), "bitinformation_wrapper.jl"
    )
    Main.path = path_to_julia_functions
    jl.using("BitInformation")
    jl.using("Pkg")
    jl.eval("include(Main.path)")


def bit_partitioning(dtype):
    if dtype.kind == "f":
        n_bits = np.finfo(dtype).bits
        n_sign = 1
        n_exponent = np.finfo(dtype).nexp
        n_mantissa = np.finfo(dtype).nmant
    elif dtype.kind == "i":
        n_bits = np.iinfo(dtype).bits
        n_sign = 1
        n_exponent = 0
        n_mantissa = n_bits - n_sign
    elif dtype.kind == "u":
        n_bits = np.iinfo(dtype).bits
        n_sign = 0
        n_exponent = 0
        n_mantissa = n_bits - n_sign
    else:
        raise ValueError(f"dtype {dtype} neither known nor implemented.")
    assert (
        n_sign + n_exponent + n_mantissa == n_bits
    ), "The components of the datatype could not be safely inferred."
    return n_bits, n_sign, n_exponent, n_mantissa


def get_bit_coords(dtype):
    """Get coordinates for bits based on dtype."""
    n_bits, n_sign, n_exponent, n_mantissa = bit_partitioning(dtype)
    coords = (
        n_sign * ["±"]
        + [f"e{int(i)}" for i in range(1, n_exponent + 1)]
        + [f"m{int(i)}" for i in range(1, n_mantissa + 1)]
    )
    return coords


def dict_to_dataset(info_per_bit):
    """Convert keepbits dictionary to :py:class:`xarray.Dataset`."""
    dsb = xr.Dataset()
    for v in info_per_bit.keys():
        dtype = np.dtype(info_per_bit[v]["dtype"])
        dim = info_per_bit[v]["dim"]
        dim_name = f"bit{dtype}"
        dsb[v] = xr.DataArray(
            info_per_bit[v]["bitinfo"],
            dims=[dim_name],
            coords={dim_name: get_bit_coords(dtype), "dim": dim},
            name=v,
            attrs={
                "long_name": f"{v} bitwise information",
                "units": 1,
            },
        ).astype("float64")
    # add metadata
    dsb.attrs = {
        "xbitinfo_description": "bitinformation calculated by xbitinfo.get_bitinformation wrapping bitinformation.jl",
        "python_repository": "https://github.com/observingClouds/xbitinfo",
        "julia_repository": "https://github.com/milankl/BitInformation.jl",
        "reference_paper": "http://www.nature.com/articles/s43588-021-00156-2",
        "xbitinfo_version": __version__,
        "BitInformation.jl_version": get_julia_package_version("BitInformation"),
    }
    for c in dsb.coords:
        if "bit" in c:
            dsb.coords[c].attrs = {
                "description": "name of the bits: '±' refers to the sign bit, 'e' to the exponents bits and 'm' to the mantissa bits."
            }
    dsb.coords["dim"].attrs = {
        "description": "dimension of the source dataset along which the bitwise information has been analysed."
    }
    return dsb


def _check_bitinfo_kwargs(implementation=None, axis=None, dim=None, kwargs=None):
    if kwargs is None:
        kwargs = {}
    # check keywords
    if implementation == "julia" and not julia_installed:
        raise ImportError('Please install julia or use implementation="python".')
    if axis is not None and dim is not None:
        raise ValueError("Please provide either `axis` or `dim` but not both.")
    if axis:
        if not isinstance(axis, int):
            raise ValueError(f"Please provide `axis` as `int`, found {type(axis)}.")
    if dim:
        if not isinstance(dim, str) and not isinstance(dim, list):
            raise ValueError(
                f"Please provide `dim` as `str` or `list`, found {type(dim)}."
            )
    if "mask" in kwargs:
        raise ValueError(
            "`xbitinfo` does not wrap the mask argument. Mask your xr.Dataset with NaNs instead."
        )
    return


def get_bitinformation(
    ds,
    dim=None,
    axis=None,
    label=None,
    overwrite=False,
    implementation="julia",
    **kwargs,
):
    """Wrap `BitInformation.jl.bitinformation() <https://github.com/milankl/BitInformation.jl/blob/main/src/mutual_information.jl>`__.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
      Input dataset to analyse
    dim : str or list
      Dimension over which to apply mean. Only one of the ``dim`` and ``axis`` arguments can be supplied.
      If no ``dim`` or ``axis`` is given (default), the bitinformation is retrieved along all dimensions.
    axis : int
      Axis over which to apply mean. Only one of the ``dim`` and ``axis`` arguments can be supplied.
      If no ``dim`` or ``axis`` is given (default), the bitinformation is retrieved along all dimensions.
    label : str
      Label of the json to serialize bitinfo. When string, serialize results to disk into file ``{{label}}.json`` to be reused later. Defaults to ``None``.
    overwrite : bool
      If ``False``, try using serialized bitinfo based on label; if true or label does not exist, run bitinformation
    implementation : str
      Bitinformation algorithm implementation. Valid options are
        - julia, the original implementation of julia in julia by Milan Kloewer
        - python, a copy of the core functionality of julia in python
    kwargs
      to be passed to bitinformation:

        - masked_value: defaults to ``NaN`` (different to ``julia`` defaulting to ``"nothing"``), set ``None`` disable masking
        - mask: use ``masked_value`` instead
        - set_zero_insignificant (``bool``): defaults to ``True`` (julia implementation) or ``False`` (python implementation)
        - confidence (``float``): defaults to ``0.99``


    Returns
    -------
    info_per_bit : :py:class:`xarray.Dataset`
      Information content per ``bit`` and ``variable`` (and ``dim``)

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> xb.get_bitinformation(ds, dim="lon")  # doctest: +ELLIPSIS
    <xarray.Dataset> Size: 652B
    Dimensions:     (bitfloat32: 32)
    Coordinates:
      * bitfloat32  (bitfloat32) <U3 384B '±' 'e1' 'e2' 'e3' ... 'm21' 'm22' 'm23'
        dim         <U3 12B 'lon'
    Data variables:
        air         (bitfloat32) float64 256B 0.0 0.0 0.0 ... 3.953e-05 0.0006889
    Attributes:
        xbitinfo_description:       bitinformation calculated by xbitinfo.get_bit...
        python_repository:          https://github.com/observingClouds/xbitinfo
        julia_repository:           https://github.com/milankl/BitInformation.jl
        reference_paper:            http://www.nature.com/articles/s43588-021-001...
        xbitinfo_version:           ...
        BitInformation.jl_version:  ...
    >>> xb.get_bitinformation(ds)
    <xarray.Dataset> Size: 1kB
    Dimensions:     (bitfloat32: 32, dim: 3)
    Coordinates:
      * bitfloat32  (bitfloat32) <U3 384B '±' 'e1' 'e2' 'e3' ... 'm21' 'm22' 'm23'
      * dim         (dim) <U4 48B 'lat' 'lon' 'time'
    Data variables:
        air         (dim, bitfloat32) float64 768B 0.0 0.0 ... 6.327e-06 0.0004285
    Attributes:
        xbitinfo_description:       bitinformation calculated by xbitinfo.get_bit...
        python_repository:          https://github.com/observingClouds/xbitinfo
        julia_repository:           https://github.com/milankl/BitInformation.jl
        reference_paper:            http://www.nature.com/articles/s43588-021-001...
        xbitinfo_version:           ...
        BitInformation.jl_version:  ...
    """
    if overwrite is False and label is not None:
        try:
            info_per_bit = load_bitinformation(label)
            info_per_bit = dict_to_dataset(info_per_bit)
        except FileNotFoundError:
            logging.info(
                f"No bitinformation could be found for {label}. Please set `overwrite=True` for recalculation..."
            )
    else:
        _check_bitinfo_kwargs(implementation, axis, dim, kwargs)
        if dim is None and axis is None:
            # gather bitinformation on all axis
            info_per_bit, label = _get_bitinformation_along_dims(
                ds,
                dim=dim,
                label=label,
                implementation=implementation,
                **kwargs,
            )
        elif isinstance(dim, list) and axis is None:
            # gather bitinformation on dims specified
            info_per_bit, label = _get_bitinformation_along_dims(
                ds,
                dim=dim,
                label=label,
                implementation=implementation,
                **kwargs,
            )
        else:
            # gather bitinformation along one axis
            info_per_bit = _get_bitinformation_along_axis(
                ds, implementation, axis, dim, **kwargs
            )
        if label is not None:
            out_fn = label + ".json"
            if not os.path.exists(out_fn) or overwrite:
                save_bitinformation(info_per_bit.to_dict(), out_fn)
        for var in info_per_bit.data_vars:  # keep attrs from input with source_ prefix
            for a in ds[var].attrs.keys():
                info_per_bit[var].attrs["source_" + a] = ds[var].attrs[a]
    return info_per_bit


def _jl_get_bitinformation(ds, var, axis, dim, kwargs={}):
    X = ds[var].values
    Main.X = X
    if axis is not None:
        # in julia convention axis + 1
        axis_jl = axis + 1
        dim = ds[var].dims[axis]
    if isinstance(dim, str):
        try:
            # in julia convention axis + 1
            axis_jl = ds[var].get_axis_num(dim) + 1
        except ValueError:
            logging.info(f"Variable {var} does not have dimension {dim}. Skipping.")
            return
    assert isinstance(axis_jl, int)
    Main.dim = axis_jl
    kwargs_str = _get_bitinformation_kwargs_handler(ds[var], kwargs)
    logging.debug(f"get_bitinformation(X, dim={dim}, {kwargs_str})")
    info_per_bit = {}
    info_per_bit["bitinfo"] = jl.eval(
        f"get_bitinformation(X, dim={axis_jl}, {kwargs_str})"
    )
    info_per_bit["dim"] = dim
    info_per_bit["axis"] = axis_jl - 1
    info_per_bit["dtype"] = str(ds[var].dtype)
    return info_per_bit


def _py_get_bitinformation(ds, var, axis, dim, kwargs={}):
    if "set_zero_insignificant" in kwargs.keys():
        if kwargs["set_zero_insignificant"]:
            raise NotImplementedError(
                "set_zero_insignificant is not implemented in the python implementation"
            )
    else:
        assert (
            kwargs == {}
        ), "This implementation only supports the plain bitinfo implementation"
    itemsize = ds[var].dtype.itemsize
    astype = f"u{itemsize}"
    X = da.array(ds[var])

    # signed exponent conversion only for floats
    if X.dtype in (np.float16, np.float32, np.float64):
        X = pb.signed_exponent(X)

    X = X.astype(astype)
    if axis is not None:
        dim = ds[var].dims[axis]
    if isinstance(dim, str):
        try:
            axis = ds[var].get_axis_num(dim)
        except ValueError:
            logging.info(f"Variable {var} does not have dimension {dim}. Skipping.")
            return
    info_per_bit = {}
    logging.info("Calling python implementation now")
    info_per_bit["bitinfo"] = pb.bitinformation(X, axis=axis).compute()
    info_per_bit["dim"] = dim
    info_per_bit["axis"] = axis
    info_per_bit["dtype"] = str(ds[var].dtype)
    return info_per_bit


def _get_bitinformation_along_dims(
    ds,
    dim=None,
    label=None,
    implementation="julia",
    **kwargs,
):
    """Helper function for :py:func:`xbitinfo.xbitinfo.get_bitinformation` to handle multi-dimensional analysis for each dim specified.

    Simple wrapper around :py:func:`xbitinfo.xbitinfo.get_bitinformation`, which calls :py:func:`xbitinfo.xbitinfo.get_bitinformation`
    for each dimension found in the provided :py:func:`xarray.Dataset`. The retrieved bitinformation
    is gathered in a joint :py:func:`xarray.Dataset` and is returned.
    """
    info_per_bit_per_dim = {}
    if dim is None:
        dim = ds.dims
    for d in dim:
        logging.info(f"Get bitinformation along dimension {d}")
        if label is not None:
            label = "_".join([label, d])
        info_per_bit_per_dim[d] = _get_bitinformation_along_axis(
            ds,
            dim=d,
            axis=None,
            implementation=implementation,
            **kwargs,
        ).expand_dims("dim", axis=0)
    info_per_bit = xr.merge(info_per_bit_per_dim.values()).squeeze()
    return info_per_bit, label


def _get_bitinformation_along_axis(ds, implementation, axis, dim, **kwargs):
    """
    Helper function for :py:func:`xbitinfo.xbitinfo.get_bitinformation` to handle analysis along one axis.
    """
    info_per_bit = {}
    pbar = tqdm(ds.data_vars)
    for var in pbar:
        pbar.set_description(f"Processing var: {var} for dim: {dim}")
        if implementation == "julia":
            info_per_bit_var = _jl_get_bitinformation(ds, var, axis, dim, kwargs)
            if info_per_bit_var is None:
                continue
            else:
                info_per_bit[var] = info_per_bit_var
        elif implementation == "python":
            info_per_bit_var = _py_get_bitinformation(ds, var, axis, dim, kwargs)
            if info_per_bit_var is None:
                continue
            else:
                info_per_bit[var] = info_per_bit_var
        else:
            raise ValueError(
                f"Implementation of bitinformation algorithm {implementation} is unknown. Please choose a different one."
            )
        info_per_bit = dict_to_dataset(info_per_bit)
    return info_per_bit


def _get_bitinformation_kwargs_handler(da, kwargs):
    """Helper function to preprocess kwargs args of :py:func:`xbitinfo.xbitinfo.get_bitinformation`."""
    kwargs_var = kwargs.copy()
    if "masked_value" not in kwargs_var:
        kwargs_var["masked_value"] = f"convert({str(da.dtype).capitalize()},NaN)"
    elif kwargs_var["masked_value"] is None:
        kwargs_var["masked_value"] = "nothing"
    if "set_zero_insignificant" not in kwargs_var:
        kwargs_var["set_zero_insignificant"] = True
    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs_var.items()])
    # convert python to julia bool
    kwargs_str = kwargs_str.replace("True", "true").replace("False", "false")
    return kwargs_str


def load_bitinformation(label):
    """Load bitinformation from JSON file"""
    label_file = label + ".json"
    if os.path.exists(label_file):
        with open(label_file) as f:
            logging.debug(f"Load bitinformation from {label+'.json'}")
            info_per_bit = json.load(f)
        return dict_to_dataset(info_per_bit)
    else:
        raise FileNotFoundError(f"No bitinformation could be found at {label+'.json'}")


def save_bitinformation(info_per_bit, out_fn, overwrite=False):
    """Save bitinformation to JSON file"""
    with open(out_fn, "w") as f:
        logging.debug(f"Save bitinformation to {out_fn}")
        json.dump(info_per_bit, f, cls=JsonCustomEncoder)
    return


def get_keepbits(info_per_bit, inflevel=0.99):
    """Get the number of mantissa bits to keep. To be used in :py:func:`xbitinfo.bitround.xr_bitround` and :py:func:`xbitinfo.bitround.jl_bitround`.

    Parameters
    ----------
    info_per_bit : :py:class:`xarray.Dataset`
      Information content of each bit. This is the output from :py:func:`xbitinfo.xbitinfo.get_bitinformation`.
    inflevel : float or list
      Level of information that shall be preserved.

    Returns
    -------
    keepbits : dict
      Number of mantissa bits to keep per variable

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> xb.get_keepbits(info_per_bit)
    <xarray.Dataset> Size: 28B
    Dimensions:   (inflevel: 1)
    Coordinates:
        dim       <U3 12B 'lon'
      * inflevel  (inflevel) float64 8B 0.99
    Data variables:
        air       (inflevel) int64 8B 7
    >>> xb.get_keepbits(info_per_bit, inflevel=0.99999999)
    <xarray.Dataset> Size: 28B
    Dimensions:   (inflevel: 1)
    Coordinates:
        dim       <U3 12B 'lon'
      * inflevel  (inflevel) float64 8B 1.0
    Data variables:
        air       (inflevel) int64 8B 14
    >>> xb.get_keepbits(info_per_bit, inflevel=1.0)
    <xarray.Dataset> Size: 28B
    Dimensions:   (inflevel: 1)
    Coordinates:
        dim       <U3 12B 'lon'
      * inflevel  (inflevel) float64 8B 1.0
    Data variables:
        air       (inflevel) int64 8B 23
    >>> info_per_bit = xb.get_bitinformation(ds)
    >>> xb.get_keepbits(info_per_bit)
    <xarray.Dataset> Size: 80B
    Dimensions:   (dim: 3, inflevel: 1)
    Coordinates:
      * dim       (dim) <U4 48B 'lat' 'lon' 'time'
      * inflevel  (inflevel) float64 8B 0.99
    Data variables:
        air       (dim, inflevel) int64 24B 5 7 6
    """
    if not isinstance(inflevel, list):
        inflevel = [inflevel]
    keepmantissabits = []
    inflevel = xr.DataArray(inflevel, dims="inflevel", coords={"inflevel": inflevel})
    if (inflevel < 0).any() or (inflevel > 1.0).any():
        raise ValueError("Please provide `inflevel` from interval [0.,1.]")
    for bitdim in [
        "bitfloat16",
        "bitfloat32",
        "bitfloat64",
        "bitint16",
        "bitint32",
        "bitint64",
        "bituint16",
        "bituint32",
        "bituint64",
    ]:
        # get only variables of bitdim
        bit_vars = [v for v in info_per_bit.data_vars if bitdim in info_per_bit[v].dims]
        if bit_vars != []:
            cdf = _cdf_from_info_per_bit(info_per_bit[bit_vars], bitdim)
            data_type = np.dtype(bitdim.replace("bit", ""))
            n_bits, _, _, n_mant = bit_partitioning(data_type)
            bitdim_non_mantissa_bits = n_bits - n_mant

            keepmantissabits_bitdim = (
                (cdf > inflevel).argmax(bitdim) + 1 - bitdim_non_mantissa_bits
            )
            # keep all mantissa bits for 100% information
            if 1.0 in inflevel:
                bitdim_all_mantissa_bits = n_bits - bitdim_non_mantissa_bits
                keepall = xr.ones_like(keepmantissabits_bitdim.sel(inflevel=1.0)) * (
                    bitdim_all_mantissa_bits
                )
                keepmantissabits_bitdim = xr.concat(
                    [keepmantissabits_bitdim.drop_sel(inflevel=1.0), keepall],
                    "inflevel",
                )
            keepmantissabits.append(keepmantissabits_bitdim)
    keepmantissabits = xr.merge(keepmantissabits)
    if inflevel.inflevel.size > 1:  # restore original ordering
        keepmantissabits = keepmantissabits.sel(inflevel=inflevel.inflevel)
    return keepmantissabits


def _cdf_from_info_per_bit(info_per_bit, bitdim):
    """Convert info_per_bit to cumulative distribution function on dimension bitdim."""
    # set below rounding error from last digit to zero
    info_per_bit_cleaned = info_per_bit.where(
        info_per_bit > info_per_bit.isel({bitdim: slice(-4, None)}).max(bitdim) * 1.5
    )
    # make cumulative distribution function
    cdf = info_per_bit_cleaned.cumsum(bitdim) / info_per_bit_cleaned.cumsum(
        bitdim
    ).isel({bitdim: -1})
    return cdf


def _jl_bitround(X, keepbits):
    """Wrap `BitInformation.jl.round <https://github.com/milankl/BitInformation.jl/blob/main/src/round_nearest.jl>`__. Used in :py:func:`xbitinfo.bitround.jl_bitround`."""
    if not julia_installed:
        raise ImportError("Please install julia or use xr_bitround")
    Main.X = X
    Main.keepbits = keepbits
    return jl.eval("round!(X, keepbits)")


def get_prefect_flow(paths=[]):
    """
    Create `prefect.Flow <https://docs.prefect.io/core/concepts/flows.html#overview>`__ for paths to be:

    1. Analyse bitwise real information content with :py:func:`xbitinfo.xbitinfo.get_bitinformation`
    2. Retrieve keepbits with :py:func:`xbitinfo.xbitinfo.get_keepbits`
    3. Apply bitrounding with :py:func:`xbitinfo.bitround.xr_bitround`
    4. Save as compressed netcdf with :py:class:`xbitinfo.save_compressed.ToCompressed_Netcdf`

    Many parameters can be changed when running the flow ``flow.run(parameters=dict(chunk="auto"))``:
    - paths: list of paths
        Paths to be bitrounded
    - analyse_paths: str or int
        Which paths to be passed to :py:func:`xbitinfo.xbitinfo.get_bitinformation`. Choose from ``["first_last", "all", int]``, where int is interpreted as stride, i.e. paths[::stride]. Defaults to ``"first"``.
    - enforce_dtype : str or None
        Enforce dtype for all variables. Currently, :py:func:`xbitinfo.xbitinfo.get_bitinformation` fails for different dtypes in variables. Do nothing if ``None``. Defaults to ``None``.
    - label : see :py:func:`xbitinfo.xbitinfo.get_bitinformation`
    - dim/axis : see :py:func:`xbitinfo.xbitinfo.get_bitinformation`
    - inflevel : see :py:func:`xbitinfo.xbitinfo.get_keepbits`
    - non_negative_keepbits : bool
        Set negative keepbits from :py:func:`xbitinfo.xbitinfo.get_keepbits` to ``0``. Required when using :py:func:`xbitinfo.bitround.xr_bitround`. Defaults to True.
    - chunks : see :py:meth:`xarray.open_mfdataset`. Note that with ``chunks=None``, ``dask`` is not used for I/O and the flow is still parallelized when using ``DaskExecutor``.
    - bitround_in_julia : bool
        Use :py:func:`xbitinfo.bitround.jl_bitround` instead of :py:func:`xbitinfo.bitround.xr_bitround`. Both should yield identical results. Defaults to ``False``.
    - overwrite : bool
        Whether to overwrite bitrounded netcdf files. ``False`` (default) skips existing files.
    - complevel : see to_compressed_netcdf, defaults to ``7``.
    - rename : list
        Replace mapping for paths towards new_path of bitrounded file, i.e. ``replace=[".nc", "_bitrounded_compressed.nc"]``

    Parameters
    ------
    paths : list
      List of paths of files to be processed by :py:func:`xbitinfo.xbitinfo.get_bitinformation`, :py:func:`xbitinfo.xbitinfo.get_keepbits`, :py:func:`xbitinfo.bitround.xr_bitround` and ``to_compressed_netcdf``.

    Returns
    -------
    prefect.Flow
      See https://docs.prefect.io/core/concepts/flows.html#overview

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

    with Flow("xbitinfo_pipeline") as flow:
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
        elif isinstance(obj, complex):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        return json.JSONEncoder.default(self, obj)


def get_julia_package_version(package):
    """Get version information of julia package"""
    if julia_installed:
        version = jl.eval(
            f'Pkg.TOML.parsefile(joinpath(pkgdir({package}), "Project.toml"))["version"]'
        )
    else:
        version = "implementation='python'"
    return version
