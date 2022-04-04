import Pkg
Pkg.add(Pkg.PackageSpec(;name="NetCDF", version="0.11.3"))
Pkg.add(["PyCall","PyPlot","Statistics","StatsBase", "JSON3", "LaTeXStrings", "JSON","BitInformation"])
Pkg.status()
