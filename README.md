<h1 align="center">
<img src="https://raw.githubusercontent.com/observingClouds/xbitinfo/refs/heads/main/docs/_static/xbitinfo_logo.svg" width="300">
</h1><br>

# xbitinfo: Retrieve bitwise information content and compress accordingly

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/observingClouds/xbitinfo/main?labpath=docs%2Fquick-start.ipynb) [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/https://github.com/observingClouds/xbitinfo/blob/main/docs/quick-start.ipynb) [![CI](https://github.com/observingClouds/xbitinfo/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/observingClouds/xbitinfo/actions/workflows/ci.yaml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/observingClouds/xbitinfo/main.svg)](https://results.pre-commit.ci/latest/github/observingClouds/xbitinfo/main) [![Documentation Status](https://readthedocs.org/projects/xbitinfo/badge/?version=latest)](https://xbitinfo.readthedocs.io/en/latest/) [![pypi](https://img.shields.io/pypi/v/xbitinfo.svg)](https://pypi.python.org/pypi/xbitinfo) ![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/xbitinfo)


Xbitinfo analyses datasets based on their bitwise real information content and applies lossy compression accordingly. Being based on [`xarray`](xarray.pydata.org/) it integrates seamlessly into common research workflows. Additional convienient functions help users to visualize the bitwise information content and to make informed decisions on the real information threshold that is subsequently used as the preserved precision during the compression.

Xbitinfo works in four steps:
1. Analyse the bitwise information content of a dataset
2. Decide on a threshold of real information to preserve (e.g. 99%)
3. Reduce the precision of the dataset accordingly (bitrounding)
4. Apply lossless compression (e.g. zlib, blosc, zstd) and store the dataset

To fullfill these steps, Xbitinfo relies on:
- `xarray` for handling multi-dimensional arrays and file formats (e.g. netcdf, zarr, hdf5, grib)
- `dask` for scaling to large datasets
- [`BitInformation.jl`](https://github.com/milankl/BitInformation.jl) (optional) for computing the bitwise information content based on the original Julia implementation. Continuous integration tests ensure however that the python-implementation shipped with xbitinfo result in identical results.
- `numcodecs` for a wide-range of lossless compression algorithms

Overall, the package presents a pipeline to compress (climate) datasets based on the real information content.


## How to install

`Xbitinfo` is packaged and distributed both via `PyPI` and `conda-forge` and can be installed via `pip` or `conda` respectively.

Depending on whether one wants to use the Julia implementation of the bitinformation algorithm (`BitInformation.jl`) or the native python implementation shipped with `xbitinfo`,
one might choose one installation option over the other.

### Pure-python installation (recommended)
```
pip install xbitinfo
```
or
```
conda install -c conda-forge xbitinfo-python
```

### Installation including optional Julia backend
```
conda install -c conda-forge xbitinfo
```
or
```
pip install xbitinfo  # julia needs to be installed manually
```

## How to use

```python
import xarray as xr
import xbitinfo as xb

# Load example dataset
# (requires pooch to be installed via e.g. `pip install pooch`)
example_dataset = "eraint_uvz"
ds = xr.tutorial.load_dataset(example_dataset)
# Step 1: analyze bitwise information content
bitinfo = xb.get_bitinformation(ds, dim="longitude")

# Step 2: decide on a threshold of real information to preserve (e.g. 99%)
keepbits = xb.get_keepbits(
    bitinfo, inflevel=0.99
)  # get number of mantissa bits to keep for 99% real information

# Step 3: reduce the precision of the dataset accordingly (bitrounding)
ds_bitrounded = xb.xr_bitround(
    ds, keepbits
)  # bitrounding keeping only keepbits mantissa bits

# Step 4: apply lossless compression (e.g. zlib, blosc, zstd) and store the dataset
ds_bitrounded.to_compressed_netcdf(outpath)
```

## How the science works

### Paper

Klöwer, M., Razinger, M., Dominguez, J. J., Düben, P. D., & Palmer, T. N. (2021). Compressing atmospheric data into its real information content. Nature Computational Science, 1(11), 713–724. doi: [10/gnm4jj](https://doi.org/10.1038/s43588-021-00156-2)

### Videos

- [General explanation of bitwise information content](https://www.youtube.com/watch?v=kcbOdwfskmY)
- [Xbitinfo implementation](https://zenodo.org/records/7259092)
- [Compression with Varying Information Density](https://zenodo.org/records/10066243)

### Julia Repository

[BitInformation.jl](https://github.com/milankl/BitInformation.jl)



## Credits

- [Milan Klöwer](https://github.com/milankl) for [BitInformation.jl](https://github.com/milankl/BitInformation.jl)
- [`Cookiecutter`](https://github.com/audreyr/cookiecutter) and [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage)
