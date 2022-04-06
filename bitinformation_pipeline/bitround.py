import xarray as xr
from numcodecs.bitround import BitRound


def bitround(data, keepbits):
    """Bitround for Arrays."""
    codec = BitRound(keepbits=keepbits)
    data = data.copy()  # otherwise overwrites the input
    encoded = codec.encode(data)
    return codec.decode(encoded)


def xr_bitround(ds, keepbits):
    """Apply bitrounding based on keepbits from bp.get_keepbits for xarray.Dataset or xr.DataArray.

    Inputs
    ------
    ds : xr.Dataset
      input netcdf to bitround
    keepbits : int or dict
      how many mantissa bits to keep

    Returns
    -------
    ds_bitrounded : xr.Dataset

    Example
    -------
        >>> ds = xr.tutorial.load_dataset("rasm")
        >>> info_per_bit = bp.get_bitinformation(ds, dim="x")
        >>> keepbits = bp.get_keepbits(ds, info_per_bit, 0.99)
        >>> ds_bitrounded = xr_bitround(ds, keepbits)
    """
    ds_bitrounded = ds.copy()
    if isinstance(ds, xr.Dataset):
        for v in ds.data_vars:
            if (
                ds[v].dtype == "float64"
            ):  # fails otherwise see https://github.com/zarr-developers/numcodecs/blob/7c7dc7cc83db1ae5c9fd93ece863acedbbc8156f/numcodecs/bitround.py#L23
                ds[v] = ds[v].astype("float32")
            if isinstance(keepbits, int):
                keep = keepbits
            elif isinstance(keepbits, dict):
                if v in keepbits.keys():
                    keep = keepbits[v]
                else:
                    continue
            # fails for .data
            ds_bitrounded[v].values = bitround(ds[v].values, keep)
            ds_bitrounded[v].attrs["bitround_keepbits"] = keepbits[v]
    elif isinstance(ds, xr.DataArray):
        if isinstance(keepbits, int):
            keep = keepbits
        elif isinstance(keepbits, dict):
            v = ds.name
            if v in keepbits.keys():
                keep = keepbits[v]
            else:
                raise ValueError("name not for in keepbits:", keepbits.keys())
        ds_bitrounded.data = bitround(ds.data, keep)
        ds_bitrounded.attrs["bitround_keepbits"] = keep
    return ds_bitrounded
