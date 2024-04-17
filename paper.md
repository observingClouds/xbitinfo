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

WIP

# Statement of need

WIP

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from ...

# References
