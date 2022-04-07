import xarray as xr
from numcodecs.bitround import BitRound

from .bitinformation_pipeline import _jl_bitround


def _bitround(data, keepbits):
    """Bitround for Arrays."""
    codec = BitRound(keepbits=keepbits)
    data = data.copy()  # otherwise overwrites the input
    encoded = codec.encode(data)
    return codec.decode(encoded)


def xr_bitround(da, keepbits):
    """Apply bitrounding based on keepbits from bp.get_keepbits for xarray.Dataset or xr.DataArray wrapping numcodecs.bitround

    Inputs
    ------
    da : xr.DataArray or xr.Dataset
      input netcdf to bitround with dtype float32
    keepbits : int or dict of {str: int}
      how many bits to keep. int

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
            da_bitrounded[v] = xr_bitround(da_bitrounded[v], keepbits)
        return da

    assert da.dtype == "float32"
    if isinstance(keepbits, int):
        keep = keepbits
    elif isinstance(keepbits, dict):
        v = da.name
        if v in keepbits.keys():
            keep = keepbits[v]
        else:
            raise ValueError(f"name {v} not for in keepbits: {keepbits.keys()}")
    # fails for .data
    da.values = _bitround(da.values, keep)
    da.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] = keep
    return da


def jl_bitround(da, keepbits):
    """Apply bitrounding based on keepbits from bp.get_keepbits for xarray.Dataset or xr.DataArray wrapping BitInformation.jl.round.

    Inputs
    ------
    da : xr.DataArray or xr.Dataset
      input netcdf to bitround with dtype float32
    keepbits : int or dict of {str: int}
      how many bits to keep. int

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
            da_bitrounded[v] = jl_bitround(da_bitrounded[v], keepbits)
        return da_bitrounded

    if isinstance(keepbits, int):
        keep = keepbits
    elif isinstance(keepbits, dict):
        v = da.name
        if v in keepbits.keys():
            keep = keepbits[v]
        else:
            raise ValueError(f"name {v} not for in keepbits: {keepbits.keys()}")
    # fails for .data
    da.values = _jl_bitround(da.values, keep)
    da.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] = keep
    return da
