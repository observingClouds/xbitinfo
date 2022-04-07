import xarray as xr
from numcodecs.bitround import BitRound


def bitround(data, keepbits):
    """Bitround for Arrays."""
    codec = BitRound(keepbits=keepbits)
    data = data.copy()  # otherwise overwrites the input
    encoded = codec.encode(data)
    return codec.decode(encoded)


def xr_bitround(da, keepbits):
    """Apply bitrounding based on keepbits from bp.get_keepbits for xarray.Dataset or xr.DataArray.

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
        >>> info_per_bit = bp.get_bitinformation(ds, dim="x")
        >>> keepbits = bp.get_keepbits(ds, info_per_bit, 0.99)
        >>> ds_bitrounded = xr_bitround(ds, keepbits)
    """
    if isinstance(da, xr.Dataset):
        da_bitrounded = da.copy()
        for v in da.data_vars:
            da_bitrounded[v] = xr_bitround(da[v], keepbits)
        return da_bitrounded

    assert da.dtype == "float32"
    da_bitrounded = da.copy()
    if isinstance(keepbits, int):
        keep = keepbits
    elif isinstance(keepbits, dict):
        v = da.name
        if v in keepbits.keys():
            keep = keepbits[v]
        else:
            raise ValueError(f"name {v} not for in keepbits: {keepbits.keys()}")
    # fails for .data
    da_bitrounded.values = bitround(
        da.values, keep - 9
    )  # uses keep mantissa bits: [0-23]
    da_bitrounded.attrs[
        "_QuantizeBitRoundNumberOfSignificantDigits"
    ] = keep  # document keepbits
    return da_bitrounded
