from dask import is_dask_collection


def get_chunksizes(da_new, da_old, for_cdo=False, time_dim="time", chunks=None):
    """Get chunksizes for xr.DataArray for to_netcdf(encoding) from original file.
    If for_cdo, ensure time chunksize of 1 when compressed."""
    assert isinstance(da_new, xr.DataArray)
    assert isinstance(da_old, xr.DataArray)
    if chunks:
        return da_new.chunk(chunks).data.chunksize
    if for_cdo:  # keep previous chunking
        time_axis_num = da_old.get_axis_num(time_dim)
        chunksize = list(da_old.shape)
        # https://code.mpimet.mpg.de/boards/2/topics/12598
        chunksize[time_axis_num] = 1
        chunksize = tuple(chunksize)
        return chunksize
    else:
        if is_dask_collection(da_new):
            return da_old.data.chunksize
        else:
            return da_old.shape


def to_compressed_netcdf(
    ds_bitrounded,
    ds,
    path,
    shuffle=True,
    complevel=9,
    for_cdo=True,
    time_dim="time",
    chunks=None,
):
    """Save bitrounded xr.Dataset to_netcdf."""
    ds_bitrounded.to_netcdf(
        path,
        encoding={
            v: {
                "zlib": True,
                "shuffle": shuffle,
                "complevel": complevel,
                "chunksizes": get_chunksizes(
                    ds_bitrounded[v],
                    ds[v],
                    for_cdo=for_cdo,
                    time_dim=time_dim,
                    chunks=chunks,
                ),
            }
            for v in keepbits
        },
    )
