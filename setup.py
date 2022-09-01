#!/usr/bin/env python

"""The setup script."""


from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as f:
    requirements = f.read().strip().split("\n")

test_requirements = ["pytest", "pytest-lazy-fixture", "pooch", "netcdf4", "dask"]

extras_require = {
    "viz": ["matplotlib", "cmcrameri"],
    "prefect": ["prefect>=1.0.0,<2.0"],
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})
extras_require["test"] = test_requirements
extras_require["docs"] = extras_require["complete"] + [
    "sphinx",
    "sphinxcontrib-napoleon",
    "sphinx-copybutton",
    "sphinx_book_theme",
    "myst-nb",
]

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
    description="Retrieve information content and compress accordingly.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="xbitinfo",
    name="xbitinfo",
    packages=find_packages(include=["xbitinfo", "xbitinfo.*"]),
    package_data={"xbitinfo": ["*.jl"]},
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/observingClouds/xbitinfo",
    zip_safe=False,
    setup_requires=[
        "setuptools_scm",
        "setuptools>=30.3.0",
    ],
    use_scm_version={
        "write_to": "xbitinfo/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
)
