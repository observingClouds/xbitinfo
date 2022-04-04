#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    author="Hauke Schulz",
    author_email="hauke.schulz@mpimet.mpg.de",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Retrieve information content and compress accordingly.",
    entry_points={
        "console_scripts": [
            "bitinformation_pipeline=bitinformation_pipeline.cli:main",
        ],
    },
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
