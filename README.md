<h1 align="center">
<img src="/docs/_static/xbitinfo_logo.svg" width="300">
</h1><br>

# xbitinfo: Retrieve information content and compress accordingly

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/observingClouds/xbitinfo/main?labpath=docs%2Fquick-start.ipynb) [![CI](https://github.com/observingClouds/xbitinfo/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/observingClouds/xbitinfo/actions/workflows/ci.yaml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/observingClouds/xbitinfo/main.svg)](https://results.pre-commit.ci/latest/github/observingClouds/xbitinfo/main) [![Documentation Status](https://readthedocs.org/projects/xbitinfo/badge/?version=latest)](https://xbitinfo.readthedocs.io/en/latest/?badge=latest) [![pypi](https://img.shields.io/pypi/v/xbitinfo.svg)](https://pypi.python.org/pypi/xbitinfo) ![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/xbitinfo)


This is an [`xarray`](xarray.pydata.org/)-wrapper around [BitInformation.jl](https://github.com/milankl/BitInformation.jl) to retrieve and apply bitrounding from within python.
The package intends to present an easy pipeline to compress (climate) datasets based on the real information content.


## How the science works

### Paper

Klöwer, M., Razinger, M., Dominguez, J. J., Düben, P. D., & Palmer, T. N. (2021). Compressing atmospheric data into its real information content. Nature Computational Science, 1(11), 713–724. doi: [10/gnm4jj](https://doi.org/10.1038/s43588-021-00156-2)

### Video

[![Video](https://img.youtube.com/vi/kcbOdwfskmY/0.jpg)](https://www.youtube.com/watch?v=kcbOdwfskmY)

### Julia Repository

[BitInformation.jl](https://github.com/milankl/BitInformation.jl)

## How to install
### Preferred installation
`conda install -c conda-forge xbitinfo`
### Alternative installation
`pip install xbitinfo` # ensure to install julia manually

## How to use

```python
import xarray as xr
import xbitinfo as xb
ds = xr.tutorial.load_dataset(inpath)
bitinfo = xb.get_bitinformation(ds, dim="lon")  # calling bitinformation.jl.bitinformation
keepbits = xb.get_keepbits(bitinfo, inflevel=0.99)  # get number of mantissa bits to keep for 99% real information
ds_bitrounded = xb.xr_bitround(ds, keepbits)  # bitrounding keeping only keepbits mantissa bits
ds_bitrounded.to_compressed_netcdf(outpath)  # save to netcdf with compression
```


## Credits

- [Milan Klöver](https://github.com/milankl) for [BitInformation.jl](https://github.com/milankl/BitInformation.jl)
- [`Cookiecutter`](https://github.com/audreyr/cookiecutter) and [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage)
