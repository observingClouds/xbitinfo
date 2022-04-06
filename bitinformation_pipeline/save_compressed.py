import xarray as xr
from dask import is_dask_collection


def get_chunksizes(da, for_cdo=False, time_dim="time", chunks=None):
    """Get chunksizes for xr.DataArray for to_netcdf(encoding) from original file.
    If for_cdo, ensure time chunksize of 1 when compressed."""
    assert isinstance(da, xr.DataArray)
    if chunks:  # use new chunksizes
        return da.chunk(chunks).data.chunksize
    if for_cdo:  # take shape as chunksize and ensure time chunksize 1
        time_axis_num = da.get_axis_num(time_dim)
        chunksize = da.data.chunksize if is_dask_collection(da) else da.shape
        # https://code.mpimet.mpg.de/boards/2/topics/12598
        chunksize = list(chunksize)
        chunksize[time_axis_num] = 1
        chunksize = tuple(chunksize)
        return chunksize
    else:
        if is_dask_collection(da):
            return da.data.chunksize
        else:
            return da.shape


def get_compress_encoding(
    ds_bitrounded,
    shuffle=True,
    complevel=9,
    for_cdo=False,
    time_dim="time",
    chunks=None,
):
    """Generate encoding for ds_bitrounded.to_netcdf(encoding).

    Example:
        >>> ds_bitrounded.to_netcdf(encoding=get_compress_encoding(ds_bitrounded))

        >>> ds_bitrounded.to_netcdf(
        ...     encoding=get_compress_encoding(ds_bitrounded, for_cdo=True)
        ... )

    """
    return {
        v: {
            "zlib": True,
            "shuffle": shuffle,
            "complevel": complevel,
            "chunksizes": get_chunksizes(
                ds_bitrounded[v], for_cdo=for_cdo, time_dim=time_dim, chunks=chunks
            ),
        }
        for v in ds_bitrounded.data_vars
    }


@xr.register_dataset_accessor("to_compressed_netcdf")
class ToCompressed_Netcdf:
    """Save to compressed netcdf wrapper.

    Example:
        >>> ds_bitrounded.to_compressed_netcdf(path)
        >>> ds_bitrounded.to_compressed_netcdf(path, complevel=4)
        >>> ds_bitrounded.to_compressed_netcdf(path, for_cdo=True)

    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(
        self,
        path,
        shuffle=True,
        complevel=9,
        for_cdo=False,
        time_dim="time",
        chunks=None,
    ):
        self._obj.to_netcdf(
            path,
            encoding=get_compress_encoding(
                self._obj,
                shuffle=shuffle,
                complevel=complevel,
                for_cdo=for_cdo,
                time_dim=time_dim,
                chunks=chunks,
            ),
        )
