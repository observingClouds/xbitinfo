import xarray as xr
from numcodecs.bitround import BitRound

from .xbitinfo import _jl_bitround, get_keepbits


def _np_bitround(data, keepbits):
    """Bitround for Arrays."""
    codec = BitRound(keepbits=keepbits)
    data = data.copy()  # otherwise overwrites the input
    encoded = codec.encode(data)
    return codec.decode(encoded)


def _keepbits_interface(da, keepbits):
    """Common interface to allowed keepbits types

    Parameters
    ----------
    da : :py:class:`xarray.DataArray`
      Input data to bitround
    keepbits : int, dict of {str: int}, :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
      How many bits to keep as int

    Returns
    -------
    keep : int
      Number of keepbits for variable given in ``da``
    """
    assert isinstance(da, xr.DataArray)
    if isinstance(keepbits, int):
        keep = keepbits
    elif isinstance(keepbits, dict):
        v = da.name
        if v in keepbits.keys():
            keep = keepbits[v]
        else:
            raise ValueError(f"name {v} not for in keepbits: {keepbits.keys()}")
    elif isinstance(keepbits, xr.Dataset):
        assert keepbits.coords["inflevel"].shape <= (
            1,
        ), "Information content is only allowed for one 'inflevel' here. Please make a selection."
        if "dim" in keepbits.coords:
            assert keepbits.coords["dim"].shape <= (
                1,
            ), "Information content is only allowed along one dimension here. Please select one `dim`. To find the maximum keepbits, simply use `keepbits.max(dim='dim')`"
        v = da.name
        if v in keepbits.keys():
            keep = int(keepbits[v])
        else:
            raise ValueError(f"name {v} not for in keepbits: {keepbits.keys()}")
    elif isinstance(keepbits, xr.DataArray):
        assert keepbits.coords["inflevel"].shape <= (
            1,
        ), "Information content is only allowed for one 'inflevel' here. Please make a selection."
        assert keepbits.coords["dim"].shape <= (
            1,
        ), "Information content is only allowed along one dimension here. Please select one `dim`. To find the maximum keepbits, simply use `keepbits.max(dim='dim')`"
        v = da.name
        if v == keepbits.name:
            keep = int(keepbits)
        else:
            raise KeyError(f"no keepbits found for variable {v}")
    else:
        raise TypeError(f"type {type(keepbits)} is not a valid type for keepbits.")
    return keep


def xr_bitround(da, keepbits):
    """Apply bitrounding based on keepbits from :py:func:`xbitinfo.xbitinfo.get_keepbits` for :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray` wrapping ``numcodecs.bitround``

    Parameters
    ----------
    da : :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
      Input data to bitround
    keepbits : int, dict of {str: int}, :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
      How many bits to keep as int. Fails if dict or :py:class:`xarray.Dataset` and key or variable not present.

    Returns
    -------
    da_bitrounded : :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> keepbits = xb.get_keepbits(info_per_bit, 0.99)
    >>> ds_bitrounded = xb.xr_bitround(ds, keepbits)
    """
    if isinstance(da, xr.Dataset):
        da_bitrounded = da.copy()
        for v in da.data_vars:
            da_bitrounded[v] = xr_bitround(da[v], keepbits)
        return da_bitrounded

    assert isinstance(da, xr.DataArray)
    keep = _keepbits_interface(da, keepbits)

    da = xr.apply_ufunc(_np_bitround, da, keep, dask="parallelized", keep_attrs=True)
    da.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] = keep
    return da


def jl_bitround(da, keepbits):
    """Apply bitrounding based on keepbits from :py:func:`xbitinfo.xbitinfo.get_keepbits` for :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray` wrapping `BitInformation.jl.round <https://github.com/milankl/BitInformation.jl/blob/main/src/round_nearest.jl>`__.

    Parameters
    ----------
    da : :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
      Input data to bitround
    keepbits : int, dict of {str: int}, :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
      How many bits to keep as int. Fails if dict or :py:class:`xarray.Dataset` and key or variable not present.

    Returns
    -------
    da_bitrounded : :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> keepbits = xb.get_keepbits(info_per_bit, 0.99)
    >>> ds_bitrounded = xb.jl_bitround(ds, keepbits)
    """
    if isinstance(da, xr.Dataset):
        da_bitrounded = da.copy()
        for v in da.data_vars:
            da_bitrounded[v] = jl_bitround(da[v], keepbits)
        return da_bitrounded

    assert isinstance(da, xr.DataArray)
    keep = _keepbits_interface(da, keepbits)
    da = xr.apply_ufunc(_jl_bitround, da, keep, dask="forbidden", keep_attrs=True)
    da.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] = keep
    return da


def bitround_along_dim(
    ds, info_per_bit, dim, inflevels=[1.0, 0.9999, 0.99, 0.975, 0.95]
):
    """
    Apply bitrounding on slices along dim based on inflevels.
    Helper function to generate data for Fig. 3 in Klöwer et al. 2021.

    Klöwer, M., Razinger, M., Dominguez, J. J., Düben, P. D., & Palmer, T. N. (2021).
    Compressing atmospheric data into its real information content.
    Nature Computational Science, 1(11), 713–724. doi: 10/gnm4jj

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`, :py:class:`xarray.DataArray`
      Input
    info_per_bit : dict
      Information content of each bit for each variable in ds. This is the output from get_bitinformation.
    dim : str
      Name of dimension for slicing
    inflevels : list of floats
      Level of information that shall be preserved. Defaults to ``[1.0, 0.9999, 0.99, 0.975, 0.95]``.

    Returns
    -------
    ds : :py:class:`xarray.Dataset`, :py:class:`xarray.DataArray`
      Bitrounded on slices along ``dim`` based on ``inflevels``

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> ds_bitrounded_along_lon = xb.bitround.bitround_along_dim(
    ...     ds, info_per_bit, dim="lon"
    ... )
    >>> (ds - ds_bitrounded_along_lon)["air"].isel(time=0).plot()  # doctest: +ELLIPSIS
    <matplotlib.collections.QuadMesh object at ...>
    """
    stride = ds[dim].size // len(inflevels)
    new_ds = []
    for i, inf in enumerate(inflevels):  # last slice might be a bit larger
        ds_slice = ds.isel(
            {
                dim: slice(
                    stride * i, stride * (i + 1) if i != len(inflevels) - 1 else None
                )
            }
        )
        keepbits_slice = get_keepbits(info_per_bit, inf)
        if inf != 1:
            ds_slice_bitrounded = xr_bitround(ds_slice, keepbits_slice)
        else:
            ds_slice_bitrounded = ds_slice
        new_ds.append(ds_slice_bitrounded)
    return xr.concat(new_ds, dim)
