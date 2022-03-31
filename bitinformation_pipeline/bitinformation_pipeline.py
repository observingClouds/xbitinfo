from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Main  # noqa: E402

jl.eval('import Pkg; Pkg.add("BitInformation"); Pkg.add("NetCDF");')
jl.eval('Pkg.add("PyPlot"), Pkg.add("StatsBase"); Pkg.add("ColorSchemes");')
jl.using("BitInformation")

jl.eval('include("get_n_plot_bitinformation.jl")')
Main.inputfile = "EUREC4A_DOM01_radiation_20200122T180000Z_latlon.nc"
bitinformation = jl.eval("get_bitinformation(inputfile)")

print(bitinformation)
