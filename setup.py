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

with open("CHANGELOG.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as f:
    requirements = f.read().strip().split("\n")

test_requirements = ["pytest", "pooch", "netcdf4", "dask"]

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
    keywords="xbitinfo",
    name="xbitinfo",
    packages=find_packages(include=["xbitinfo", "xbitinfo.*"]),
    package_data={"xbitinfo": ["*.jl"]},
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/observingClouds/xbitinfo",
    zip_safe=False,
    use_scm_version={"version_scheme": "post-release", "local_scheme": "dirty-tag"},
    setup_requires=[
        "setuptools_scm",
        "setuptools>=30.3.0",
        "setuptools_scm_git_archive",
    ],
)
