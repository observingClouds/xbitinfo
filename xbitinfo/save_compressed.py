import logging

import numcodecs
import xarray as xr


def get_chunksizes(da, for_cdo=False, time_dim="time", chunks=None):
    """Get chunksizes for :py:class:`xarray.DataArray` for ``to_netcdf(encoding)`` from original file.
    If ``for_cdo=True``, ensure ``time_dim`` ``chunksize`` of 1 when compressed."""
    assert isinstance(da, xr.DataArray)
    if chunks:  # use new chunksizes
        return da.chunk(chunks).data.chunksize
    if for_cdo:  # take shape as chunksize and ensure time chunksize 1
        if time_dim in da.dims:
            time_axis_num = da.get_axis_num(time_dim)
            chunksize = da.data.chunksize if da.chunks is not None else da.shape
            # https://code.mpimet.mpg.de/boards/2/topics/12598
            chunksize = list(chunksize)
            chunksize[time_axis_num] = 1
            chunksize = tuple(chunksize)
            return chunksize
        else:
            return get_chunksizes(da, for_cdo=False, time_dim=time_dim)
    else:
        return da.data.chunksize if da.chunks is not None else da.shape


def get_compress_encoding_nc(
    ds_bitrounded,
    compression="zlib",
    shuffle=True,
    complevel=9,
    for_cdo=False,
    time_dim="time",
    chunks=None,
):
    """Generate encoding for :py:meth:`xarray.Dataset.to_netcdf`.

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("rasm")
    >>> get_compress_encoding_nc(ds)
    {'Tair': {'zlib': True, 'shuffle': True, 'complevel': 9, 'chunksizes': (36, 205, 275)}}
    >>> get_compress_encoding_nc(ds, for_cdo=True)
    {'Tair': {'zlib': True, 'shuffle': True, 'complevel': 9, 'chunksizes': (1, 205, 275)}}

    See also
    --------
    :py:meth:`xarray.Dataset.to_netcdf`

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
    """Save to compressed ``netcdf`` wrapping :py:meth:`xarray.Dataset.to_netcdf` with :py:func:`xbitinfo.save_compressed.get_compress_encoding_nc`.

    Parameters
    ----------
    path : str, path-like or file-like
      Path to which to save this dataset
    compression : str
      Compression library used for encoding. Defaults to ``"zlib"``.
    shuffle : bool
      Netcdf shuffle used for encoding. Defaults to ``True``.
    complevel : int
      Compression level used for encoding.
      Ranges from 2 (little compression, fast) to 9 (strong compression, slow). Defaults to ``7``.
    for_cdo : bool
      If you want to continue working with ``cdo``. If ``True``, sets time chunksize to 1,
      context https://code.mpimet.mpg.de/boards/2/topics/12598. Defaults to ``False``.
    time_dim : str
      Name of the time dimension. Defaults to ``"time"``.
    chunks : str, dict
      How should the data be chunked on disk. None keeps defaults. ``"auto"`` uses ``dask.chunk("auto")``,
      dict individual chunking. Defaults to ``None``.
    kwargs : dict
      Kwargs to be passed to :py:meth:`xarray.Dataset.to_netcdf`

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("rasm")
    >>> path = "compressed_rasm.nc"
    >>> ds.to_compressed_netcdf(path)
    >>> ds.to_compressed_netcdf(path, complevel=4)
    >>> ds.to_compressed_netcdf(path, for_cdo=True)

    See also
    --------
    :py:meth:`xarray.Dataset.to_netcdf`

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
        **kwargs,
    ):
        self._obj.to_netcdf(
            path,
            encoding=get_compress_encoding_nc(
                self._obj,
                compression=compression,
                shuffle=shuffle,
                complevel=complevel,
                for_cdo=for_cdo,
                time_dim=time_dim,
                chunks=chunks,
            ),
            **kwargs,
        )


def get_compress_encoding_zarr(
    ds_bitrounded,
    compressor=numcodecs.Blosc("zstd", shuffle=numcodecs.Blosc.BITSHUFFLE),
):
    """Generate encoding for :py:meth:`xarray.Dataset.to_zarr`.

    Example
    -------
        >>> ds = xr.tutorial.load_dataset("rasm")
        >>> get_compress_encoding_zarr(ds)
        {'Tair': {'compressor': Blosc(cname='zstd', clevel=5, shuffle=BITSHUFFLE, blocksize=0)}}

    See also
    --------
    :py:meth:`xarray.Dataset.to_zarr`
    """
    encoding = {}
    if isinstance(compressor, dict):
        for v in ds_bitrounded.data_vars:
            if v in compressor.keys():
                encoding[v] = {"compressor": compressor[v]}
            else:
                logging.warning(
                    f"No compressor given for variable {v}. Using default compressor"
                )
                encoding[v] = {
                    "compressor": numcodecs.Blosc(
                        "zstd", shuffle=numcodecs.Blosc.BITSHUFFLE
                    )
                }
    else:
        for v in ds_bitrounded.data_vars:
            encoding[v] = {"compressor": compressor}

    return encoding


@xr.register_dataset_accessor("to_compressed_zarr")
class ToCompressed_Zarr:
    """Save to compressed ``zarr`` wrapping :py:meth:`xarray.Dataset.to_zarr` with :py:func:`xbitinfo.save_compressed.get_compress_encoding_zarr`.

    Parameters
    ----------
    path : str, path-like or file-like
      Output location of compressed dataset
    compressor : numcodecs
      Compressor used for encoding. Defaults to zstd with bit-shuffling.
    kwargs : dict
      Arguments to be passed to :py:meth:`xarray.Dataset.to_zarr`

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("rasm")
    >>> path = "compressed_rasm.zarr"
    >>> ds.to_compressed_zarr(path, mode="w")
    >>> ds.to_compressed_zarr(path, compressor=numcodecs.Blosc("zlib"), mode="w")
    >>> ds.to_compressed_zarr(
    ...     path, compressor={"Tair": numcodecs.Blosc("zstd")}, mode="w"
    ... )

    See also
    --------
    :py:meth:`xarray.Dataset.to_zarr`

    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(
        self,
        path,
        compressor=numcodecs.Blosc("zstd", shuffle=numcodecs.Blosc.BITSHUFFLE),
        **kwargs,
    ):
        self._obj.to_zarr(
            path,
            encoding=get_compress_encoding_zarr(
                self._obj,
                compressor=compressor,
            ),
            **kwargs,
        )
