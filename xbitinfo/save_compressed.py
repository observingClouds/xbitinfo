import numcodecs
import xarray as xr
from zarr.codecs import BloscCodec, BloscShuffle


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
    ds,
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
    >>> ds = xr.Dataset({"Tair": (("time", "x", "y"), np.random.rand(36, 20, 10))})
    >>> get_compress_encoding_nc(ds)
    {'Tair': {'zlib': True, 'shuffle': True, 'complevel': 9, 'chunksizes': (36, 20, 10)}}
    >>> get_compress_encoding_nc(ds, for_cdo=True)
    {'Tair': {'zlib': True, 'shuffle': True, 'complevel': 9, 'chunksizes': (1, 20, 10)}}

    See also
    --------
    :py:meth:`xarray.Dataset.to_netcdf`

    """
    enc_checker = xr.backends.netCDF4_._extract_nc4_variable_encoding
    return {
        v: {
            **enc_checker(ds[v]),
            compression: True,
            "shuffle": shuffle,
            "complevel": complevel,
            "chunksizes": get_chunksizes(
                ds[v], for_cdo=for_cdo, time_dim=time_dim, chunks=chunks
            ),
        }
        for v in ds.data_vars
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
        engine="netcdf4",
        **kwargs,
    ):
        assert engine == "netcdf4", "Only 'netcdf4' engine is currently supported."
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
            engine=engine,
            **kwargs,
        )


def get_compress_encoding_zarr(
    ds,
    compressor=BloscCodec(cname="zstd", shuffle=BloscShuffle.bitshuffle),
    zarr_format="2",
):
    """Generate encoding for :py:meth:`xarray.Dataset.to_zarr`.

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("rasm")
    >>> get_compress_encoding_zarr(ds)  # doctest: +NORMALIZE_WHITESPACE
    {'Tair': {'chunks': None,
              'compressor': BloscCodec(typesize=None,
                                       cname=<BloscCname.zstd: 'zstd'>,
                                       clevel=5,
                                       shuffle=<BloscShuffle.bitshuffle: 'bitshuffle'>,
                                       blocksize=0)}}

    See also
    --------
    :py:meth:`xarray.Dataset.to_zarr`
    """
    encoding = {}
    enc_checker = xr.backends.zarr.extract_zarr_variable_encoding
    if zarr_format == "2":
        compressor_key = "compressor"
    elif zarr_format == "3":
        compressor_key = "compressors"
    if isinstance(compressor, dict):
        default_compressor = BloscCodec(cname="zstd", shuffle=BloscShuffle.bitshuffle)
        encoding = {
            v: {
                **enc_checker(ds[v], zarr_format=zarr_format),
                compressor_key: compressor.get(v, default_compressor),
            }
            for v in ds.data_vars
        }
    else:
        encoding = {
            v: {
                **enc_checker(ds[v], zarr_format=zarr_format),
                compressor_key: compressor,
            }
            for v in ds.data_vars
        }

    # chunks 'auto' can cause issues
    for v, enc in encoding.items():
        chunks = enc.get("chunks", None)
        if chunks is not None:
            if chunks == "auto":
                enc["chunks"] = None

    return encoding


@xr.register_dataset_accessor("to_compressed_zarr")
class ToCompressed_Zarr:
    """Save to compressed ``zarr`` wrapping :py:meth:`xarray.Dataset.to_zarr` with :py:func:`xbitinfo.save_compressed.get_compress_encoding_zarr`.

    Parameters
    ----------
    path : str, path-like or file-like
      Output location of compressed dataset
    compressor : BloscCodec
      Compressor used for encoding. Defaults to zstd with bit-shuffling.
    kwargs : dict
      Arguments to be passed to :py:meth:`xarray.Dataset.to_zarr`

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("rasm")
    >>> path = "compressed_rasm.zarr"
    >>> ds.to_compressed_zarr(path, mode="w")
    >>> ds.to_compressed_zarr(path, compressor=BloscCodec(cname="zlib"), mode="w")
    >>> ds.to_compressed_zarr(
    ...     path, compressor={"Tair": BloscCodec(cname="zstd")}, mode="w"
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
        compressor=BloscCodec(cname="zstd", shuffle=BloscShuffle.bitshuffle),
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
