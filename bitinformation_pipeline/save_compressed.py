import xarray as xr
from dask import is_dask_collection


def get_chunksizes(da, for_cdo=False, time_dim="time", chunks=None):
    """Get chunksizes for xr.DataArray for to_netcdf(encoding) from original file.
    If for_cdo, ensure time chunksize of 1 when compressed."""
    assert isinstance(da, xr.DataArray)
    if chunks:  # use new chunksizes
        return da.chunk(chunks).data.chunksize
    if for_cdo:  # take shape as chunksize and ensure time chunksize 1
        if time_dim in da.dims:
            time_axis_num = da.get_axis_num(time_dim)
            chunksize = da.data.chunksize if is_dask_collection(da) else da.shape
            # https://code.mpimet.mpg.de/boards/2/topics/12598
            chunksize = list(chunksize)
            chunksize[time_axis_num] = 1
            chunksize = tuple(chunksize)
            return chunksize
        else:
            return get_chunksizes(da, for_cdo=False, time_dim=time_dim)
    else:
        return da.data.chunksize if is_dask_collection(da.data) else da.shape


def get_compress_encoding(
    ds_bitrounded,
    compression="zlib",
    shuffle=True,
    complevel=9,
    for_cdo=False,
    time_dim="time",
    chunks=None,
):
    """Generate encoding for ds_bitrounded.to_netcdf(encoding).

    Example:
        >>> ds = xr.tutorial.load_dataset("rasm")
        >>> get_compress_encoding(ds)
        {'Tair': {'zlib': True, 'shuffle': True, 'complevel': 9, 'chunksizes': (36, 205, 275)}}
        >>> get_compress_encoding(ds, for_cdo=True)
        {'Tair': {'zlib': True, 'shuffle': True, 'complevel': 9, 'chunksizes': (1, 205, 275)}}

    """
    return {
        v: {
            compression: True,
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
    """Save to compressed netcdf wrapping ds.to_netcdf(encoding=get_compress_encoding(ds)).

    Inputs
    ------
    path : str, path-like or file-like
      Path to which to save this dataset
    compression : str
      compression library, used for encoding. Defaults to "zlib".
    shuffle : bool
      netcdf shuffle, used for encording. Defaults to True.
    complevel : int
      compression level, used for encoding.
      Ranges for 2 (little compression, fast) to 9 (strong compression, slow). Defaults to 7.
    for_cdo : bool
      Continue working with cdo. If True, sets time chunksize to 1,
      context https://code.mpimet.mpg.de/boards/2/topics/12598. Defaults to False.
    time_dim : str
      name of the time dimension. Defaults to "time".
    chunks : str, dict
      how should the data be chunked on disk. None keeps defaults. "auto" uses dask.chunk("auto"),
      dict individual chunking. Defaults to None.
    kwargs : dict
      to be passed to xr.Dataset.to_netcdf(**kwargs)

    Example:
        >>> ds = xr.tutorial.load_dataset("rasm")
        >>> path = "compressed_rasm.nc"
        >>> ds.to_compressed_netcdf(path)
        >>> ds.to_compressed_netcdf(path, complevel=4)
        >>> ds.to_compressed_netcdf(path, for_cdo=True)

    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(
        self,
        path,
        compression="zlib",
        shuffle=True,
        complevel=9,
        for_cdo=False,
        time_dim="time",
        chunks=None,
        **kwargs
    ):
        self._obj.to_netcdf(
            path,
            encoding=get_compress_encoding(
                self._obj,
                compression=compression,
                shuffle=shuffle,
                complevel=complevel,
                for_cdo=for_cdo,
                time_dim=time_dim,
                chunks=chunks,
            ),
            **kwargs
        )
