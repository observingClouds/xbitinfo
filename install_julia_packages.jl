import Pkg
Pkg.add(["PyCall","PyPlot","Statistics","StatsBase", "JSON3", "LaTXStrings", "JSON","BitInformation"])
Pkg.add(Pkg.PackageSpec(;name="NetCDF", version="0.11.3"))
Pkg.status()
