import Pkg
Pkg.add(Pkg.PackageSpec(;name="NetCDF", version="0.11.0"))
Pkg.add(["PyCall","PyPlot","Statistics","StatsBase", "JSON3", "LaTeXStrings", "JSON","BitInformation"])
Pkg.status()
