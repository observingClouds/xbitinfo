# bitinformation_pipeline

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/observingClouds/bitinformation_pipeline/main) [![CI](https://github.com/observingClouds/bitinformation_pipeline/actions/workflows/ci.yaml/badge.svg)](https://github.com/observingClouds/bitinformation_pipeline/actions/workflows/ci.yaml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/observingClouds/bitinformation_pipeline/main.svg)](https://results.pre-commit.ci/latest/github/observingClouds/bitinformation_pipeline/main)

Retrieve information content and compress accordingly.

This will be a wrapper around [BitInformation.jl](https://github.com/milankl/BitInformation.jl) to retrieve and apply bitrounding from within python.
The package intends to present an easy pipeline to compress (climate) datasets based on the real information content.


## How the science works

### Paper

Klöwer, M., Razinger, M., Dominguez, J. J., Düben, P. D., & Palmer, T. N. (2021). Compressing atmospheric data into its real information content. Nature Computational Science, 1(11), 713–724. doi: [10/gnm4jj](https://doi.org/10.1038/s43588-021-00156-2)

### Video

[![Video](https://img.youtube.com/vi/kcbOdwfskmY/0.jpg)](https://www.youtube.com/watch?v=kcbOdwfskmY)

### Julia Repository

[BitInformation.jl](https://github.com/milankl/BitInformation.jl)

## How to install

`pip install git+https://github.com/observingClouds/bitinformation_pipeline.git`

## How to use

```python
import xarray as xr
import bitinformation_pipeline as bp
ds = xr.tutorial.load_dataset(inpath)
bitinfo = bp.get_bitinformation(ds, dim="lon")  # calling bitinformation.jl.bitinformation
keepbits = bp.get_keepbits(bitinfo, inflevel=0.99)  # get number of mantissa bits to keep for 99% real information
ds_bitrounded = bp.xr_bitround(ds, keepbits)  # bitrounding keeping only keepbits mantissa bits
ds_bitrounded.to_compressed_netcdf(outpath)  # save to netcdf with compression
```

see [quick-start.ipynb](https://nbviewer.org/github/observingClouds/bitinformation_pipeline/blob/main/examples/quick-start.ipynb)

## Credits

- [Milan Klöver](https://github.com/milankl) for [BitInformation.jl](https://github.com/milankl/BitInformation.jl)
- [`Cookiecutter`](https://github.com/audreyr/cookiecutter) and [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage)
