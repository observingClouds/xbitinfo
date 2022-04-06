#!/usr/bin/env python

"""The setup script."""

import os

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

julia_install_command = "julia install_julia_packages.jl"


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        os.system(julia_install_command)


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        os.system(julia_install_command)


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["xarray", "julia"]

test_requirements = ["pytest", "pooch", "netcdf4", "git+https://github.com/observingClouds/numcodecs@bitround"]

setup(
    author="Hauke Schulz",
    author_email="hauke.schulz@mpimet.mpg.de",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    description="Retrieve information content and compress accordingly.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="bitinformation_pipeline",
    name="bitinformation_pipeline",
    packages=find_packages(
        include=["bitinformation_pipeline", "bitinformation_pipeline.*"]
    ),
    package_data={"bitinformation_pipeline": ["*.jl"]},
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/observingClouds/bitinformation_pipeline",
    version="0.0.1",
    zip_safe=False,
)
