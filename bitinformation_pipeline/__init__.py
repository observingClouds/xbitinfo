"""Top-level package for bitinformation_pipeline."""

__author__ = """Hauke Schulz"""
__email__ = "hauke.schulz@mpimet.mpg.de"
__version__ = "0.0.1"

from .bitround import xr_bitround

from .bitinformation_pipeline import (
    get_bitinformation,
    get_keepbits,
    plot_bitinformation,
)
