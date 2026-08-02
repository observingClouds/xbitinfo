import json
import logging
import os
import warnings

import numpy as np
import xarray as xr
from dask import array as da

try:
    from prefect import flow, task, unmapped

    prefect_import_error = None
except ImportError as e:
    flow = task = unmapped = None
    prefect_import_error = e

try:
    from julia.api import Julia

    julia_installed = True
except ImportError:
    julia_installed = False
from tqdm.auto import tqdm

import xbitinfo as xb

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


def _check_for_nans(ds):
    """Warn if any variable in *ds* contains NaN values.

    BitInformation results can be inaccurate or undefined when the input
    contains NaN values.  This pre-flight check issues one UserWarning per
    affected variable so the caller is aware before any expensive computation
    starts.

    To suppress::

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="xbitinfo")
    """
    for var in ds.data_vars:
        nan_count = int(ds[var].isnull().sum())
        if nan_count > 0:
            warnings.warn(
                f"Variable {var!r} contains {nan_count} NaN value(s). "
                "BitInformation results may be inaccurate for data with NaNs. "
                "Consider masking or filling NaNs before calling get_bitinformation(), "
                "or pass masked_value=None to disable NaN masking explicitly.",
                UserWarning,
                stacklevel=3,
            )


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
    dsb.attrs = {
        "xbitinfo_description": "bitinformation calculated by xbitinfo.get_bitinformation wrapping bitinformation.jl",
        "python_repository": "https://github.com/observingClouds/xbitinfo",
        "julia_repository": "https://github.com/milankl/BitInformation.jl",
        "reference_paper": "http://www.nature.com/articles/s43588-021-00156-2",
        "xbitinfo_version": xb.__version__,
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
    """Wrap `BitInformation.jl.bitinformation()`.

    See full docstring in the original source. This wrapper adds a NaN
    pre-flight check (issue #200) before delegating to the core implementation.
    """
    # Warn the user early if the dataset contains NaN values so they are not
    # surprised by inaccurate bitinfo results (issue #200).
    _check_for_nans(ds)

    if overwrite is False and label is not None:
        try:
            info_per_bit = load_bitinformation(label)
        except FileNotFoundError:
            logging.info(
                f"No bitinformation could be found for {label}. Please set `overwrite=True` for recalculation..."
            )
        else:
            return info_per_bit
    else:
        _check_bitinfo_kwargs(implementation, axis, dim, kwargs)

    return _get_bitinformation(
        ds,
        dim=dim,
        axis=axis,
        label=label,
        overwrite=overwrite,
        implementation=implementation,
        **kwargs,
    )


def _get_bitinformation(
    ds,
    dim=None,
    axis=None,
    label=None,
    overwrite=False,
    implementation="julia",
    **kwargs,
):
    if dim is None and axis is None:
        return _get_bitinformation_along_dims(
            ds,
            dim=dim,
            label=label,
            overwrite=overwrite,
            implementation=implementation,
            **kwargs,
        )
    if isinstance(dim, list) and axis is None:
        return _get_bitinformation_along_dims(
            ds,
            dim=dim,
            label=label,
            overwrite=overwrite,
            implementation=implementation,
            **kwargs,
        )
    else:
        info_per_bit = _get_bitinformation_along_axis(
            ds, implementation, axis, dim, **kwargs
        )

        if label is not None:
            out_fn = label + ".json"
            if not os.path.exists(out_fn) or overwrite:
                save_bitinformation(info_per_bit, out_fn)

        info_per_bit = dict_to_dataset(info_per_bit)

    for var in info_per_bit.data_vars:
        for a in ds[var].attrs.keys():
            info_per_bit[var].attrs["source_" + a] = ds[var].attrs[a]
    return info_per_bit


def _quantized_variable_is_scaled(ds, var):
    has_scale_or_offset = any(
        ["add_offset" in ds[var].encoding, "scale_factor" in ds[var].encoding]
    )
    if not has_scale_or_offset:
        return False
    loaded_dtype = ds[var].dtype
    storage_dtype = ds[var].encoding.get("dtype", None)
    assert (
        storage_dtype is not None
    ), f"Variable {var} is likely quantized, but does not have a storage dtype"
    if loaded_dtype == storage_dtype:
        return False
    return True


def _jl_get_bitinformation(ds, var, axis, dim, kwargs={}):
    X = ds[var].values
    Main.X = X
    if axis is not None:
        axis_jl = axis + 1
        dim = ds[var].dims[axis]
    if isinstance(dim, str):
        try:
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
    overwrite=False,
    implementation="julia",
    **kwargs,
):
    info_per_bit_per_dim = {}
    if dim is None:
        dim = ds.dims
    for d in dim:
        logging.info(f"Get bitinformation along dimension {d}")
        if label is not None:
            label = "_".join([label, d])
        info_per_bit_per_dim[d] = _get_bitinformation(
            ds,
            dim=d,
            axis=None,
            label=label,
            overwrite=overwrite,
            implementation=implementation,
            **kwargs,
        ).expand_dims("dim", axis=0)
    info_per_bit = xr.merge(
        info_per_bit_per_dim.values(), join="outer", compat="no_conflicts"
    ).squeeze()
    return info_per_bit


def _get_bitinformation_along_axis(ds, implementation, axis, dim, **kwargs):
    info_per_bit = {}
    pbar = tqdm(ds.data_vars)
    for var in pbar:
        pbar.set_description(f"Processing var: {var} for dim: {dim}")
        if _quantized_variable_is_scaled(ds, var):
            loaded_dtype = ds[var].dtype
            quantized_storage_dtype = ds[var].encoding["dtype"]
            warnings.warn(
                f"Variable {var} is quantized as {quantized_storage_dtype}, but loaded as {loaded_dtype}. Consider reopening using `mask_and_scale=False` to get sensible results",
                category=UserWarning,
            )
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
    return info_per_bit


def _get_bitinformation_kwargs_handler(da, kwargs):
    kwargs_var = kwargs.copy()
    if "masked_value" not in kwargs_var:
        if da.dtype.kind == "i" or da.dtype.kind == "u":
            logging.warning(
                "No masked value given for integer type variable. Assuming no mask to apply."
            )
            kwargs_var["masked_value"] = "nothing"
        elif da.dtype.kind == "f":
            kwargs_var["masked_value"] = f"convert({str(da.dtype).capitalize()},NaN)"
        else:
            raise ValueError(f"Dtype kind ({da.dtype.kind}) not supported.")
    elif kwargs_var["masked_value"] is None:
        kwargs_var["masked_value"] = "nothing"
    if "set_zero_insignificant" not in kwargs_var:
        kwargs_var["set_zero_insignificant"] = True
    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs_var.items()])
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


def get_keepbits(info_per_bit, inflevel=0.99, information_filter=None, **kwargs):
    """Get the number of mantissa bits to keep."""
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
        bit_vars = [v for v in info_per_bit.data_vars if bitdim in info_per_bit[v].dims]
        if bit_vars != []:
            if information_filter == "Gradient":
                cdf = get_cdf_without_artificial_information(
                    info_per_bit[bit_vars],
                    bitdim,
                    kwargs["threshold"],
                    kwargs["tolerance"],
                    bit_vars,
                )
            else:
                cdf = _cdf_from_info_per_bit(info_per_bit[bit_vars], bitdim)
            data_type = np.dtype(bitdim.replace("bit", ""))
            n_bits, _, _, n_mant = bit_partitioning(data_type)
            bitdim_non_mantissa_bits = n_bits - n_mant
            keepmantissabits_bitdim = (
                (cdf > inflevel).argmax(bitdim) + 1 - bitdim_non_mantissa_bits
            )
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
    keepmantissabits = xr.merge(keepmantissabits, join="outer", compat="no_conflicts")
    if inflevel.inflevel.size > 1:
        keepmantissabits = keepmantissabits.sel(inflevel=inflevel.inflevel)
    return keepmantissabits


def _cdf_from_info_per_bit(info_per_bit, bitdim):
    info_per_bit_cleaned = info_per_bit.where(
        info_per_bit > info_per_bit.isel({bitdim: slice(-4, None)}).max(bitdim) * 1.5
    )
    cdf = info_per_bit_cleaned.cumsum(bitdim) / info_per_bit_cleaned.cumsum(
        bitdim
    ).isel({bitdim: -1})
    return cdf


def get_cdf_without_artificial_information(
    info_per_bit, bitdim, threshold, tolerance, bit_vars
):
    """Calculate CDF with artificial information removal."""
    coordinates = info_per_bit.coords
    coordinates_array = coordinates["dim"].values
    flag_scalar_value = False
    if coordinates_array.ndim == 0:
        value = coordinates_array.item()
        flag_scalar_value = True
        coordinates_array = np.array([value])

    cdf = _cdf_from_info_per_bit(info_per_bit, bitdim)
    for var_name in bit_vars:
        for dimension in coordinates_array:
            if flag_scalar_value:
                infoArray = info_per_bit[var_name]
            else:
                infoArray = info_per_bit[var_name].sel(dim=dimension)

            infSum = sum(infoArray).item()
            data_type = np.dtype(bitdim.replace("bit", ""))
            _, n_sign, n_exponent, _ = bit_partitioning(data_type)
            sign_and_exponent = n_sign + n_exponent
            SignExpSum = sum(infoArray[:sign_and_exponent]).item()

            if flag_scalar_value:
                cdf_array = cdf[var_name]
            else:
                cdf_array = cdf[var_name].sel(dim=dimension)

            gradient_array = np.diff(cdf_array.values)
            CurrentBit_Sum = SignExpSum
            for i in range(sign_and_exponent, len(gradient_array) - 1):
                CurrentBit_Sum = CurrentBit_Sum + infoArray[i].item()
                if (
                    gradient_array[i]
                ) < tolerance and CurrentBit_Sum >= threshold * infSum:
                    infbits = i
                    break

            for i in range(0, infbits + 1):
                cdf_array[i] = cdf_array[i] / cdf_array[infbits]

            cdf_array[(infbits + 1) :] = 1
    return cdf


def _jl_bitround(X, keepbits):
    if not julia_installed:
        raise ImportError("Please install julia or use xr_bitround")
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
        elif isinstance(obj, bytes):
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
