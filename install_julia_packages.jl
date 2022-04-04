import Pkg
Pkg.add(Pkg.PackageSpec(;name="NetCDF"))
Pkg.add(["PyCall","PyPlot","Statistics","StatsBase", "JSON3", "LaTeXStrings", "JSON","BitInformation"])
Pkg.status()
