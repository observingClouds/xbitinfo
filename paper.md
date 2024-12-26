---
title: 'xbitinfo: Compressing geospatial data based on information theory'
tags:
  - Python
authors:
  - name: Hauke Schulz
    orcid: 0000-0001-5468-1137
    corresponding: true
    affiliation: "1, 2"
  - name: Milan Klöwer
    orcid: 0000-0002-3920-4356
    affiliation: 3
  - name: Aaron Spring
    affiliation: 3
affiliations:
 - name: University of Washington, Seattle, USA
   index: 1
 - name: eScience Institute, University of Washington, Seattle, USA
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 16 April 2024
bibliography: paper.bib
---

# Summary

Xbitinfo combines the workflow elements needed to analyse datasets based on their information content and to compress accordingly in one software package. Xbitinfo provides additional tools to visualize information histograms and to make informed decisions on the information threshold. Being based on xarray Datasets it allows to interact with a range of common input and output dataformats, including all numcodes compression algorithms.

# Statement of need

The geospatial field as all other fields are generating more and more data, while storage solutions have not increased at the same pace. In addition more and more data is stored in the cloud and egress fees and network speeds are more and more of a concern. Compression algorithms can help to reduce the pressure on these components significantly and are therefore commonly used.

Lossless compressions like zstd and zip are able to reduce storage consumptions without removing a single bit. Often this very conservative behaviour is however unnecessary because not all the bits are needed for the later usecase of the data or even meaningfull due to e.g. hardware limitations and numerical rounding errors.

Lossy compressions are therefore often used to remove the unnecessary bits with JPEG being a very promintent example of such compressor.

JPEG uses a perceptual model of the human eye to decide on whether or not to keep information. While this approach is acceptable for the publication of a scientific figure, it is not for the original data that still undergoes mathematical operations, like gradients. Such operations require a mathematically stable reduction in information content.

Linear quantization and logarithmic quantization are commonly used with geospatial data, not the least because it is the standard algorithm shipped with the GRIB format.

The issue with quantizations are however....

- abitrary thresholds
- thresholds should depend on the variable, high level, ...
- as a consequence lack of information or too much unneeded information stored

@klower_compressing_2021 has developed an algorithm that can destinguish between real and artificial information content based on information theory. It further allows to set a threshold for the real information content that shall be preserved in case additional compression is needed.

As typical for lossy-compressions, parameters can be set to influence the loss. In case of the bitinformation algorithm, the `inflevel` parameter can be set to decide on the percentage of real information content to be preserved. after applying the bitinformation-informed bitrounding. The compression can therefore be split into three main stages:

 - **Bitinformation**: analysing the bitinformation content
 - **Bitrounding**:
    - deciding on information content to keep (inflevel)
    - translate `inflevel` to bits to keep (`keepbits`) after rounding
    - bitrounding according to keepbits
 - **Compression**:
    - applying (lossless) compression

All stages are shown in \autoref{fig:general_workflow}.
![General workflow.\label{fig:general_workflow}](general_workflow.png){ width=40% }

One can set the `inflevel` and use implementation offered by CDO (>=v2.1.0), numcodecs (>=v) or the Julia implementation provided by @klower_compressing_2021. However, in practice the decision on how much information shall be kept needs testing with the downstream tools and is often an iterative process to ensure consistent behaviour with the original dataset. The gathering of the bitinformation and the decision on the bitrounding parameters are therefore often not immediately following each other and are interrupted by visual inspection and testing (see \autoref{fig:xbitinfo_workflow})

![Xbitinfo workflow with the addition of storing the computational expensive retrieval of the bitinformation content in a JSON file for later reference and the ability to evaluate and adjust the keepbits on subsets of the original dataset.\label{fig:xbitinfo_workflow}](xbitinfo_workflow.png){ width=40% }

Xbitinfo therefore provides additional convience functions over  @klower_compressing_2021 to analyse, filter and visualize the information content. Because Xbitinfo operates on xarray `Datasets` it can also handle a large variety of input and output formats, like netCDF and Zarr and naturally fit into several current scientific workflows. Thanks to the xarray-compatibility it can also make use of a wide range of modern compression algorithms that are implemented for the specific output data formats to utilize the additional compression gains due to reduced information.

Xbitinfo provides two backends for the calculation of the bitinformation content, one wraps the latest Julia implementation provided with @klower_compressing_2021 for consistency and the other uses numpy to be dask compatible and more performant.


# Example

To compress a dataset based on its bitinformation content with xbitinfo follows the following steps:

```python
import xarray as xr
import xbitinfo as xb

ds = xr.open_dataset("/path/to/input/file")
bitinfo = xb.get_bitinformation(ds)
keepbits = xb.get_keepbits(bitinfo, inflevel=0.95)
ds = xb.xr_bitround(ds, keepbits)
ds.to_compressed_zarr("/path/to/output/file")
```


# Remarks

It should be noted that the BitInformation algorithm relies on uncompressed data that hasn't been manipulated beforehand. A common issue is that climate model output has been linearily quantized during its generation, e.g. because it has been written to the GRIB format. Such datasets should be handeled with care as the bitinformation often contains artificial information resulting in too many keepbits. Filters to capture those cases are currently developed within xbitinfo to warn the user.


<!-- # Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

# Acknowledgements

We acknowledge contributions from ...

# References
