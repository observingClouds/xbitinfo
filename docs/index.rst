xbitinfo: Retrieve information content and compress accordingly
===============================================================

.. image:: https://github.com/observingClouds/xbitinfo/actions/workflows/ci.yaml/badge.svg
   :target: https://github.com/observingClouds/xbitinfo/actions/workflows/ci.yaml

.. image:: https://results.pre-commit.ci/badge/github/observingClouds/xbitinfo/main.svg
   :target: https://results.pre-commit.ci/latest/github/observingClouds/xbitinfo/main

.. image:: https://img.shields.io/pypi/v/xbitinfo.svg
   :target: https://pypi.python.org/pypi/xbitinfo/

.. image:: https://img.shields.io/readthedocs/bitinfo/latest.svg?style=flat
   :target: https://xbitinfo.readthedocs.io/en/latest

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/observingClouds/xbitinfo/main?labpath=docs%2Fquick-start.ipynb

This is an `xarray <xarray.pydata.org>`__-wrapper around `BitInformation.jl <https://github.com/milankl/BitInformation.jl>`__ to retrieve and apply bitrounding from within python.
The package intends to present an easy pipeline to compress (climate) datasets based on the real information content.


How the science works
=====================

Paper
-----

Klöwer, M., Razinger, M., Dominguez, J. J., Düben, P. D., & Palmer, T. N. (2021). Compressing atmospheric data into its real information content. Nature Computational Science, 1(11), 713–724. doi: `10/gnm4jj <https://doi.org/10.1038/s43588-021-00156-2>`__

Video
-----

.. image:: https://img.youtube.com/vi/kcbOdwfskmY/0.jpg
   :target: https://www.youtube.com/watch?v=kcbOdwfskmY)

Julia Repository
----------------

`BitInformation.jl <https://github.com/milankl/BitInformation.jl>`__


How to install
==============

You can install the latest release of ``xbitinfo`` soon using ``pip``:

.. code-block:: bash

    pip install xbitinfo


You can also install the bleeding edge (pre-release versions) by running:

.. code-block:: bash

    pip install git+https://github.com/observingClouds/xbitinfo

    pip install git+https://github.com/observingClouds/xbitinfo.git#egg=xbitinfo[complete]



How to use
==========

.. code-block:: python

    import xarray as xr
    import xbitinfo as xb

    ds = xr.tutorial.load_dataset(inpath)
    bitinfo = xb.get_bitinformation(ds, dim="lon")  # calling bitinformation.jl.bitinformation
    keepbits = xb.get_keepbits(bitinfo, inflevel=0.99)  # get number of mantissa bits to keep for 99% real information
    ds_bitrounded = xb.xr_bitround(ds, keepbits)  # bitrounding keeping only keepbits mantissa bits
    ds_bitrounded.to_compressed_netcdf(outpath)  # save to netcdf with compression


Credits
=======

- `Milan Klöver <https://github.com/milankl>`__ for `BitInformation.jl <https://github.com/milankl/BitInformation.jl>`__
- `Cookiecutter <https://github.com/audreyr/cookiecutter>`__ and `audreyr/cookiecutter-pypackage <https://github.com/audreyr/cookiecutter-pypackage>`__


**Getting Started**

* :doc:`quick-start`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    quick-start.ipynb

**Help & Reference**

* :doc:`api`
* :doc:`contributing`
* :doc:`changelog`
* :doc:`authors`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & Reference

   api
   contributing
   changelog
   authors
