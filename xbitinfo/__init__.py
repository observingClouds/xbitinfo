"""Top-level package for xbitinfo."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .bitround import jl_bitround, xr_bitround
from .graphics import plot_bitinformation, plot_distribution
from .save_compressed import get_compress_encoding_nc, get_compress_encoding_zarr
from .xbitinfo import get_bitinformation, get_keepbits, get_prefect_flow
