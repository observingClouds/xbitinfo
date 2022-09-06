import json
import logging
import os

import numpy as np
import xarray as xr
from julia.api import Julia
from tqdm.auto import tqdm

from . import __version__
from .julia_helpers import install

already_ran = False
if not already_ran:
    already_ran = install(quiet=True)


jl = Julia(compiled_modules=False, debug=True)
from julia import Main  # noqa: E402

path_to_julia_functions = os.path.join(
    os.path.dirname(__file__), "bitinformation_wrapper.jl"
)
Main.path = path_to_julia_functions
jl.using("BitInformation")
jl.using("Pkg")
jl.eval("include(Main.path)")


NMBITS = {64: 12, 32: 9, 16: 6}  # number of non mantissa bits for given dtype


def get_bit_coords(dtype_size):
    """Get coordinates for bits assuming float dtypes."""
    if dtype_size == 16:
        coords = (
            ["±"]
            + [f"e{int(i)}" for i in range(1, 6)]
            + [f"m{int(i-5)}" for i in range(6, 16)]
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
    """Convert keepbits dictionary to :py:class:`xarray.Dataset`."""
    dsb = xr.Dataset()
    for v in info_per_bit.keys():
        dtype_size = len(info_per_bit[v]["bitinfo"])
        dim = info_per_bit[v]["dim"]
        dim_name = f"bit{dtype_size}"
        dsb[v] = xr.DataArray(
            info_per_bit[v]["bitinfo"],
            dims=[dim_name],
            coords={dim_name: get_bit_coords(dtype_size), "dim": dim},
            name=v,
            attrs={"long_name": f"{v} bitwise information", "units": "1"},
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


def get_bitinformation(ds, dim=None, axis=None, label=None, overwrite=False, **kwargs):
    """Wrap `BitInformation.jl.bitinformation() <https://github.com/milankl/BitInformation.jl/blob/main/src/mutual_information.jl>`__.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
      Input dataset to analyse
    dim : str
      Dimension over which to apply mean. Only one of the ``dim`` and ``axis`` arguments can be supplied.
      If no ``dim`` or ``axis`` is given (default), the bitinformation is retrieved along all dimensions.
    axis : int
      Axis over which to apply mean. Only one of the ``dim`` and ``axis`` arguments can be supplied.
      If no ``dim`` or ``axis`` is given (default), the bitinformation is retrieved along all dimensions.
    label : str
      Label of the json to serialize bitinfo. When string, serialize results to disk into file ``{{label}}.json`` to be reused later. Defaults to ``None``.
    overwrite : bool
      If ``False``, try using serialized bitinfo based on label; if true or label does not exist, run bitinformation
    kwargs
      to be passed to bitinformation:

        - masked_value: defaults to ``NaN`` (different to ``bitinformation.jl`` defaulting to ``"nothing"``), set ``None`` disable masking
        - mask: use ``masked_value`` instead
        - set_zero_insignificant (``bool``): defaults to ``True``
        - confidence (``float``): defaults to ``0.99``


    Returns
    -------
    info_per_bit : :py:class:`xarray.Dataset`
      Information content per ``bit`` and ``variable`` (and ``dim``)

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> xb.get_bitinformation(ds, dim="lon")  # doctest: +ELLIPSIS
    <xarray.Dataset>
    Dimensions:  (bit32: 32)
    Coordinates:
      * bit32    (bit32) <U3 '±' 'e1' 'e2' 'e3' 'e4' ... 'm20' 'm21' 'm22' 'm23'
        dim      <U3 'lon'
    Data variables:
        air      (bit32) float64 0.0 0.0 0.0 0.0 ... 0.0 3.953e-05 0.0006889
    Attributes:
        xbitinfo_description:       bitinformation calculated by xbitinfo.get_bit...
        python_repository:          https://github.com/observingClouds/xbitinfo
        julia_repository:           https://github.com/milankl/BitInformation.jl
        reference_paper:            http://www.nature.com/articles/s43588-021-001...
        xbitinfo_version:           ...
        BitInformation.jl_version:  ...
    >>> xb.get_bitinformation(ds)
    <xarray.Dataset>
    Dimensions:  (bit32: 32, dim: 3)
    Coordinates:
      * bit32    (bit32) <U3 '±' 'e1' 'e2' 'e3' 'e4' ... 'm20' 'm21' 'm22' 'm23'
      * dim      (dim) <U4 'lat' 'lon' 'time'
    Data variables:
        air      (dim, bit32) float64 0.0 0.0 0.0 0.0 ... 0.0 6.327e-06 0.0004285
    Attributes:
        xbitinfo_description:       bitinformation calculated by xbitinfo.get_bit...
        python_repository:          https://github.com/observingClouds/xbitinfo
        julia_repository:           https://github.com/milankl/BitInformation.jl
        reference_paper:            http://www.nature.com/articles/s43588-021-001...
        xbitinfo_version:           ...
        BitInformation.jl_version:  ...
    """
    if dim is None and axis is None:
        # gather bitinformation on all axis
        return _get_bitinformation_along_dims(
            ds, dim=dim, label=label, overwrite=overwrite, **kwargs
        )
    if isinstance(dim, list) and axis is None:
        # gather bitinformation on dims specified
        return _get_bitinformation_along_dims(
            ds, dim=dim, label=label, overwrite=overwrite, **kwargs
        )
    else:
        # gather bitinformation along one axis
        if overwrite is False and label is not None:
            try:
                info_per_bit = load_bitinformation(label)
                return info_per_bit
            except FileNotFoundError:
                logging.info(
                    f"No bitinformation could be found for {label}. Recalculating..."
                )

        # check keywords
        if axis is not None and dim is not None:
            raise ValueError("Please provide either `axis` or `dim` but not both.")
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
                axis_jl = axis + 1
                dim = ds[var].dims[axis]
            if isinstance(dim, str):
                try:
                    # in julia convention axis + 1
                    axis_jl = ds[var].get_axis_num(dim) + 1
                except ValueError:
                    logging.info(
                        f"Variable [var] does not have dimension {dim}. Skipping."
                    )
                    continue
            assert isinstance(axis_jl, int)
            Main.dim = axis_jl
            kwargs_str = _get_bitinformation_kwargs_handler(ds[var], kwargs)
            logging.debug(f"get_bitinformation(X, dim={dim}, {kwargs_str})")
            info_per_bit[var] = {}
            info_per_bit[var]["bitinfo"] = jl.eval(
                f"get_bitinformation(X, dim={axis_jl}, {kwargs_str})"
            )
            info_per_bit[var]["dim"] = dim
            info_per_bit[var]["axis"] = axis_jl - 1
        if label is not None:
            with open(label + ".json", "w") as f:
                logging.debug(f"Save bitinformation to {label + '.json'}")
                json.dump(info_per_bit, f, cls=JsonCustomEncoder)
    return dict_to_dataset(info_per_bit)


def _get_bitinformation_along_dims(ds, dim=None, label=None, overwrite=False, **kwargs):
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
        info_per_bit_per_dim[d] = get_bitinformation(
            ds, dim=d, axis=None, label=label, overwrite=overwrite, **kwargs
        ).expand_dims("dim", axis=0)
    info_per_bit = xr.merge(info_per_bit_per_dim.values()).squeeze()
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
    <xarray.Dataset>
    Dimensions:   (inflevel: 1)
    Coordinates:
        dim       <U3 'lon'
      * inflevel  (inflevel) float64 0.99
    Data variables:
        air       (inflevel) int64 7
    >>> xb.get_keepbits(info_per_bit, inflevel=0.99999999)
    <xarray.Dataset>
    Dimensions:   (inflevel: 1)
    Coordinates:
        dim       <U3 'lon'
      * inflevel  (inflevel) float64 1.0
    Data variables:
        air       (inflevel) int64 14
    >>> xb.get_keepbits(info_per_bit, inflevel=1.0)
    <xarray.Dataset>
    Dimensions:   (inflevel: 1)
    Coordinates:
        dim       <U3 'lon'
      * inflevel  (inflevel) float64 1.0
    Data variables:
        air       (inflevel) int64 23
    >>> info_per_bit = xb.get_bitinformation(ds)
    >>> xb.get_keepbits(info_per_bit)
    <xarray.Dataset>
    Dimensions:   (dim: 3, inflevel: 1)
    Coordinates:
      * dim       (dim) <U4 'lat' 'lon' 'time'
      * inflevel  (inflevel) float64 0.99
    Data variables:
        air       (dim, inflevel) int64 5 7 6
    """
    if not isinstance(inflevel, list):
        inflevel = [inflevel]
    keepmantissabits = []
    inflevel = xr.DataArray(inflevel, dims="inflevel", coords={"inflevel": inflevel})
    if (inflevel < 0).any() or (inflevel > 1.0).any():
        raise ValueError("Please provide `inflevel` from interval [0.,1.]")
    for bitdim in ["bit16", "bit32", "bit64"]:
        # get only variables of bitdim
        bit_vars = [v for v in info_per_bit.data_vars if bitdim in info_per_bit[v].dims]
        if bit_vars != []:
            cdf = _cdf_from_info_per_bit(info_per_bit[bit_vars], bitdim)
            bitdim_non_mantissa_bits = NMBITS[int(bitdim[3:])]
            keepmantissabits_bitdim = (
                (cdf > inflevel).argmax(bitdim) + 1 - bitdim_non_mantissa_bits
            )
            # keep all mantissa bits for 100% information
            if 1.0 in inflevel:
                bitdim_all_mantissa_bits = int(bitdim[3:]) - bitdim_non_mantissa_bits
                keepall = xr.ones_like(keepmantissabits_bitdim.sel(inflevel=1.0)) * (
                    bitdim_all_mantissa_bits
                )
                keepmantissabits_bitdim = xr.concat(
                    [keepmantissabits_bitdim.drop_sel(inflevel=1.0), keepall],
                    "inflevel",
                )
            keepmantissabits.append(keepmantissabits_bitdim)
    keepmantissabits = xr.merge(keepmantissabits)
    if inflevel.inflevel.size > 1:  # restore orginal ordering
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
        elif isinstance(obj, (complex, np.complex)):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        return json.JSONEncoder.default(self, obj)


def get_julia_package_version(package):
    """Get version information of julia package"""
    version = jl.eval(
        f'Pkg.TOML.parsefile(joinpath(pkgdir({package}), "Project.toml"))["version"]'
    )
    return version
