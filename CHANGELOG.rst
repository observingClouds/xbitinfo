=========
CHANGELOG
=========

unreleased
------------------

* Add note to docs on additional requirements running ArtificialInformation filter example (:pr:`322`) `Hauke Schulz`_.
* Improve README code example (:pr:`325`) `Hauke Schulz`_.
* Join pytest toml sections to support pytest>=9.0.0 (:pr:`326`) `Hauke Schulz`_.

0.0.5 (2025-11-07)
------------------

* Update pre-commit hooks and update doctests (:pr:`307`) `Hauke Schulz`_.
* Add python matrix test and restrict supported versions to 3.11-3.13 (:pr:`323`) `Hauke Schulz`_.

0.0.4 (2025-09-22)
------------------

* Improve documentation (:pr:`310`) `Hauke Schulz`_.
* Fix deprecation compatibility issues (:pr:`309`) `Hauke Schulz`_.
* Limit libcurl version to fix recent binary issues (:pr:`297`) `Hauke Schulz`_.
* Add warning for quantized variables (:pr:`286`, :issue:`202`) `Joel Jaeschke`_.
* Update BitInformation.jl version to v0.6.3 (:pr:`292`) `Hauke Schulz`_
* Improve test/docs environment separation (:pr:`275`, :issue:`267`) `Aryan Bakliwal`_.
* Set default masked value to None for integers (:pr:`289`) `Hauke Schulz`_.
* Add basic filter to remove artificial information from bitinformation (:pr:`280`, :issue:`209`) `Ishaan Jain`_.
* Add support for additional datatypes in :py:func:`xbitinfo.xbitinfo.plot_bitinformation` (:pr:`218`, :issue:`168`) `Hauke Schulz`_.
* Drop python 3.8 support and add python 3.11 (:pr:`175`) `Hauke Schulz`_.
* Implement basic retrieval of bitinformation in python as alternative to julia implementation (:pr:`156`, :issue:`155`, :pr:`126`, :issue:`125`) `Hauke Schulz`_ with helpful comments from `Milan Klöwer`_.
* Make julia binding to BitInformation.jl optional (:pr:`153`, :issue:`151`) `Aaron Spring`_.
* Add support for python 3.13 (:pr:`303`) and use uv for GitHub Actions `Hauke Schulz`_.
* Simplify get_bitinformation function (:pr:`262`, :issue:`261`) `Hauke Schulz`_.

0.0.3 (2022-07-11)
------------------

* Fix julia package installations for PyPi and enable installation via pip and conda (:issue:`18`, :pr:`132`, :pr:`131`) `Filipe Fernandes`_, `Mark Kittisopikul`_.
* Fix compression example for zarr-files (:issue:`119`, :pr:`121`) `Hauke Schulz`_.
* Keep ``attrs`` as ``source_attribute`` from input in :py:func:`xbitinfo.xbitinfo.get_bitinformation`. (:issue:`154`, :pr:`158`) `Aaron Spring`_.

0.0.2 (2022-07-11)
------------------

* Fix ``kwargs`` in :py:func:`xbitinfo.xbitinfo._get_bitinformation_kwargs_handler` which were not reused for other variables in :py:func:`xbitinfo.xbitinfo.get_bitinformation`.
  (:issue:`99`, :pr:`101`) `Aaron Spring`_.
* Refactor :py:func:`xbitinfo.xbitinfo.get_keepbits` with xarray functions.
  (:pr:`100`) `Aaron Spring`_.
* Allow ``dim`` as ``list`` in :py:func:`xbitinfo.xbitinfo.get_bitinformation`.
  (:issue:`105`, :pr:`106`) `Aaron Spring`_.
* Fix PyPI package and make it actually installable via pip (:issue:`14`, :pr:`114`, :pr:`103`) `Aaron Spring`_, `Hauke Schulz`_, `Rich Signell`_.
* Improve PyPi packaging (:pr:`110`)  `Filipe Fernandes`_.

0.0.1 (2022-05-04)
------------------

* First release on PyPI.
