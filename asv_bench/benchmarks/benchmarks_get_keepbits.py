import numpy as np
import xarray as xr

from xbitinfo import get_keepbits

from . import _skip_slow, ensure_loaded, parameterized, randn, requires_dask


class GetKeepbits:
    """
    Benchmark time and peak memory of `get_keepbits`.
    """

    # https://asv.readthedocs.io/en/stable/benchmarks.html
    timeout = 30.0
    repeat = 3
    number = 5

    def setup(self, *args, **kwargs):
        self.info_per_bit = {
            "air": np.array(
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    3.94447851e-01,
                    3.94447851e-01,
                    3.94447851e-01,
                    3.94447851e-01,
                    3.94447851e-01,
                    3.94310542e-01,
                    7.36739987e-01,
                    5.62682836e-01,
                    3.60511555e-01,
                    1.52471111e-01,
                    4.18818055e-02,
                    3.65276146e-03,
                    1.19975820e-05,
                    4.39366160e-05,
                    4.18329296e-05,
                    2.54572089e-05,
                    1.44121797e-04,
                    1.34144798e-03,
                    1.55468479e-06,
                    5.38601212e-04,
                    8.09862581e-04,
                    1.74893445e-04,
                    4.97915410e-05,
                    3.88027711e-04,
                    0.00000000e00,
                    3.95323228e-05,
                    6.88854435e-04,
                ]
            )
        }

    def time_get_keepbits(self, **kwargs):
        """Take time for `get_keepbits`."""
        get_keepbits(self.info_per_bit, **kwargs)

    def peakmem_get_keepbits(self, **kwargs):
        """Take memory peak for `get_keepbits`."""
        get_keepbits(self.info_per_bit, **kwargs)
