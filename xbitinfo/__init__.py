"""Top-level package for xbitinfo."""

from . import _version
from .bitround import jl_bitround, xr_bitround
from .graphics import plot_bitinformation, plot_distribution
from .save_compressed import get_compress_encoding_nc, get_compress_encoding_zarr
from .xbitinfo import _get_keepbits, get_bitinformation, get_keepbits, get_prefect_flow

__version__ = _version.get_versions()["version"]
