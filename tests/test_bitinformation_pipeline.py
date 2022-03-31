#!/usr/bin/env python

"""Tests for `bitinformation_pipeline` package."""

import pytest

# from bitinformation_pipeline import bitinformation_pipeline


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
    from bitinformation_pipeline import bitinformation_pipeline as bm
    import xarray as xr
    ds = xr.tutorial.load_dataset("rasm")
    bitinfo = bm.get_bitinformation(ds, dim=1)


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
