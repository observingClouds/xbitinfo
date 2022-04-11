import xarray as xr
from numcodecs.bitround import BitRound

from .bitinformation_pipeline import _jl_bitround


def _xr_bitround(data, keepbits):
    """Bitround for Arrays."""
    codec = BitRound(keepbits=keepbits)
    data = data.copy()  # otherwise overwrites the input
    encoded = codec.encode(data)
    return codec.decode(encoded)


def xr_bitround(da, keepbits, map_blocks=False):
    """Apply bitrounding based on keepbits from bp.get_keepbits for xarray.Dataset or xr.DataArray wrapping numcodecs.bitround

    Inputs
    ------
    da : xr.DataArray or xr.Dataset
      input netcdf to bitround with dtype float32
    keepbits : int or dict of {str: int}
      how many bits to keep. int
    map_blocks : bool
      if True, use xr.map_blocks, else use xr.apply_ufunc. Defaults to False.

    Returns
    -------
    da_bitrounded : xr.DataArray or xr.Dataset

    Example
    -------
        >>> ds = xr.tutorial.load_dataset("air_temperature")
        >>> info_per_bit = bp.get_bitinformation(ds, dim="lon")
        >>> keepbits = bp.get_keepbits(ds, info_per_bit, 0.99)
        >>> ds_bitrounded = bp.xr_bitround(ds, keepbits)
    """
    if isinstance(da, xr.Dataset):
        da_bitrounded = da.copy()
        for v in da.data_vars:
            da_bitrounded[v] = xr_bitround(da[v], keepbits, map_blocks=map_blocks)
        return da_bitrounded

    assert isinstance(da, xr.DataArray)
    assert da.dtype == "float32"
    if isinstance(keepbits, int):
        keep = keepbits
    elif isinstance(keepbits, dict):
        v = da.name
        if v in keepbits.keys():
            keep = keepbits[v]
        else:
            raise ValueError(f"name {v} not for in keepbits: {keepbits.keys()}")
    if map_blocks:
        da = da.map_blocks(_xr_bitround, args=[keep])
    else:
        da = xr.apply_ufunc(
            _xr_bitround, da, keep, dask="parallelized", keep_attrs=True
        )
    da.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] = keep
    return da


def jl_bitround(da, keepbits, map_blocks=False):
    """Apply bitrounding based on keepbits from bp.get_keepbits for xarray.Dataset or xr.DataArray wrapping BitInformation.jl.round.

    Inputs
    ------
    da : xr.DataArray or xr.Dataset
      input netcdf to bitround with dtype float32
    keepbits : int or dict of {str: int}
      how many bits to keep. int
    map_blocks : bool
      if True, use xr.map_blocks, else use xr.apply_ufunc. Defaults to False.

    Returns
    -------
    da_bitrounded : xr.DataArray or xr.Dataset

    Example
    -------
        >>> ds = xr.tutorial.load_dataset("air_temperature")
        >>> info_per_bit = bp.get_bitinformation(ds, dim="lon")
        >>> keepbits = bp.get_keepbits(ds, info_per_bit, 0.99)
        >>> ds_bitrounded = bp.jl_bitround(ds, keepbits)
    """
    if isinstance(da, xr.Dataset):
        da_bitrounded = da.copy()
        for v in da.data_vars:
            da_bitrounded[v] = jl_bitround(da[v], keepbits, map_blocks=map_blocks)
        return da_bitrounded

    if isinstance(keepbits, int):
        keep = keepbits
    elif isinstance(keepbits, dict):
        v = da.name
        if v in keepbits.keys():
            keep = keepbits[v]
        else:
            raise ValueError(f"name {v} not for in keepbits: {keepbits.keys()}")
    if map_blocks:
        da = da.map_blocks(_jl_bitround, args=[keep])
    else:
        da = xr.apply_ufunc(
            _jl_bitround, da, keep, dask="parallelized", keep_attrs=True
        )
    da.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] = keep
    return da
