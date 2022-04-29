"""Top-level package for xbitinfo."""

from pkg_resources import DistributionNotFound, get_distribution

from .bitround import jl_bitround, xr_bitround
from .graphics import plot_bitinformation, plot_distribution
from .save_compressed import get_compress_encoding_nc, get_compress_encoding_zarr
from .xbitinfo import _get_keepbits, get_bitinformation, get_keepbits, get_prefect_flow

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    __version__ = "0.0.0"  # pragma: no cover
