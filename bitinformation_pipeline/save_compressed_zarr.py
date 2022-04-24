import logging

import numcodecs
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


def get_compress_encoding_zarr(
    ds_bitrounded,
    compressor=numcodecs.Blosc("zstd", shuffle=numcodecs.Blosc.BITSHUFFLE),
):
    """Generate encoding for ds_bitrounded.to_zarr(encoding).

    Example:
        >>> ds = xr.tutorial.load_dataset("rasm")
        >>> get_compress_encoding_zarr(ds)
        {'Tair': {'zlib': True, 'shuffle': True, 'complevel': 9, 'chunksizes': (36, 205, 275)}}
        >>> get_compress_encoding(ds, for_cdo=True)
        {'Tair': {'zlib': True, 'shuffle': True, 'complevel': 9, 'chunksizes': (1, 205, 275)}}

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
    """Save to compressed zarr wrapping ds.to_zarr(encoding=get_compress_encoding_zarr(ds)).

    Inputs
    ------
    path : str, path-like or file-like
      Path to which to save this dataset
    compressor : numcodecs
      compressor used for encoding. Defaults to zstd with bit-shuffling.
    kwargs : dict
      to be passed to xr.Dataset.to_zarr(**kwargs)

    Example:
        >>> ds = xr.tutorial.load_dataset("rasm")
        >>> path = "compressed_rasm.zarr"
        >>> ds.to_compressed_zarr(path)
        >>> ds.to_compressed_zarr(path, compressor=numcodecs.Blosc("zlib"))
        >>> ds.to_compressed_zarr(path, compressor={"Tair": numcodecs.Blosc("zstd")})

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
