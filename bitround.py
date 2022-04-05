from numcodecs.bitround import BitRound


def bitround(data, keepbits):
    codec = BitRound(keepbits=keepbits)
    data = data.copy()  # otherwise overwrites the input
    encoded = codec.encode(data)
    return codec.decode(encoded)


def xr_bitround(ds, keepbits):
    """Apply bitrounding based on keepbits from bp.get_keepbits for xarray.Dataset or xr.DataArray."""
    ds_bitrounded = ds.copy()
    if isinstance(ds, xr.Dataset):
        for v in ds.data_vars:
            if v in keepbits.keys():
                # fails for .data
                if (
                    ds[v].dtype == "float64"
                ):  # fails otherwise see https://github.com/zarr-developers/numcodecs/blob/7c7dc7cc83db1ae5c9fd93ece863acedbbc8156f/numcodecs/bitround.py#L23
                    ds[v] = ds[v].astype("float32")
                ds_bitrounded[v].values = bitround(ds[v].values, keepbits[v])
                ds_bitrounded[v].attrs["bitround_keepbits"] = keepbits[v]
    elif isinstance(ds, xr.DataArray):
        if v in keepbits.keys():
            v = ds.name
            ds_bitrounded.data = bitround(ds.data, keepbits[v])
            ds_bitrounded.attrs["bitround_keepbits"] = keepbits[v]
    return ds_bitrounded
