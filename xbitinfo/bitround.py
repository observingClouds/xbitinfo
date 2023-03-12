import xarray as xr
from numcodecs.bitround import BitRound

from .xbitinfo import _jl_bitround, get_inflevels


def _np_bitround(data, inflevels):
    """Bitround for Arrays."""
    codec = BitRound(inflevels=inflevels)
    data = data.copy()  # otherwise overwrites the input
    encoded = codec.encode(data)
    return codec.decode(encoded)


def _inflevels_interface(da, inflevels):
    """Common interface to allowed inflevels types

    Parameters
    ----------
    da : :py:class:`xarray.DataArray`
      Input data to bitround
    inflevels : int, dict of {str: int}, :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
      How many bits to keep as int

    Returns
    -------
    keep : int
      Number of inflevels for variable given in ``da``
    """
    assert isinstance(da, xr.DataArray)
    if isinstance(inflevels, int):
        keep = inflevels
    elif isinstance(inflevels, dict):
        v = da.name
        if v in inflevels.keys():
            keep = inflevels[v]
        else:
            raise ValueError(f"name {v} not for in inflevels: {inflevels.keys()}")
    elif isinstance(inflevels, xr.Dataset):
        assert inflevels.coords["inflevel"].shape <= (
            1,
        ), "Information content is only allowed for one 'inflevel' here. Please make a selection."
        if "dim" in inflevels.coords:
            assert inflevels.coords["dim"].shape <= (
                1,
            ), "Information content is only allowed along one dimension here. Please select one `dim`. To find the maximum inflevels, simply use `inflevels.max(dim='dim')`"
        v = da.name
        if v in inflevels.keys():
            keep = int(inflevels[v])
        else:
            raise ValueError(f"name {v} not for in inflevels: {inflevels.keys()}")
    elif isinstance(inflevels, xr.DataArray):
        assert inflevels.coords["inflevel"].shape <= (
            1,
        ), "Information content is only allowed for one 'inflevel' here. Please make a selection."
        assert inflevels.coords["dim"].shape <= (
            1,
        ), "Information content is only allowed along one dimension here. Please select one `dim`. To find the maximum inflevels, simply use `inflevels.max(dim='dim')`"
        v = da.name
        if v == inflevels.name:
            keep = int(inflevels)
        else:
            raise KeyError(f"no inflevels found for variable {v}")
    else:
        raise TypeError(f"type {type(inflevels)} is not a valid type for inflevels.")
    return keep


def xr_bitround(da, inflevels):
    """Apply bitrounding based on inflevels from :py:func:`xbitinfo.xbitinfo.get_inflevels` for :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray` wrapping ``numcodecs.bitround``

    Parameters
    ----------
    da : :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
      Input data to bitround
    inflevels : int, dict of {str: int}, :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
      How many bits to keep as int. Fails if dict or :py:class:`xarray.Dataset` and key or variable not present.

    Returns
    -------
    da_bitrounded : :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> inflevels = xb.get_inflevels(info_per_bit, 0.99)
    >>> ds_bitrounded = xb.xr_bitround(ds, inflevels)
    """
    if isinstance(da, xr.Dataset):
        da_bitrounded = da.copy()
        for v in da.data_vars:
            da_bitrounded[v] = xr_bitround(da[v], inflevels)
        return da_bitrounded

    assert isinstance(da, xr.DataArray)
    keep = _inflevels_interface(da, inflevels)

    da = xr.apply_ufunc(_np_bitround, da, keep, dask="parallelized", keep_attrs=True)
    da.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] = keep
    return da


def jl_bitround(da, inflevels):
    """Apply bitrounding based on inflevels from :py:func:`xbitinfo.xbitinfo.get_inflevels` for :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray` wrapping `BitInformation.jl.round <https://github.com/milankl/BitInformation.jl/blob/main/src/round_nearest.jl>`__.

    Parameters
    ----------
    da : :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
      Input data to bitround
    inflevels : int, dict of {str: int}, :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
      How many bits to keep as int. Fails if dict or :py:class:`xarray.Dataset` and key or variable not present.

    Returns
    -------
    da_bitrounded : :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> inflevels = xb.get_inflevels(info_per_bit, 0.99)
    >>> ds_bitrounded = xb.jl_bitround(ds, inflevels)
    """
    if isinstance(da, xr.Dataset):
        da_bitrounded = da.copy()
        for v in da.data_vars:
            da_bitrounded[v] = jl_bitround(da[v], inflevels)
        return da_bitrounded

    assert isinstance(da, xr.DataArray)
    keep = _inflevels_interface(da, inflevels)
    da = xr.apply_ufunc(_jl_bitround, da, keep, dask="forbidden", keep_attrs=True)
    da.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] = keep
    return da


def bitround_along_dim(
    ds, info_per_bit, dim, inflevels, keepbits=[1.0, 0.9999, 0.99, 0.975, 0.95]
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
        inflevels_slice = get_inflevels(info_per_bit, inf)
        if inf != 1:
            ds_slice_bitrounded = xr_bitround(ds_slice, inflevels_slice)
        else:
            ds_slice_bitrounded = ds_slice
        new_ds.append(ds_slice_bitrounded)
    return xr.concat(new_ds, dim)
