from julia.api import Julia

jl = Julia(compiled_modules=False, debug=True)
from julia import Main  # noqa: E402

path_to_julia_functions = os.path.join(
    os.path.dirname(__file__), "get_n_plot_bitinformation.jl"
)
Main.path = path_to_julia_functions
jl.eval(
    'import Pkg; Pkg.add(["BitInformation", "NetCDF", "PyPlot", "StatsBase", "ColorSchemes"])'
)jl.using("BitInformation")

jl.eval('include("get_n_plot_bitinformation.jl")')
Main.inputfile = "EUREC4A_DOM01_radiation_20200122T180000Z_latlon.nc"
bitinformation = jl.eval("get_bitinformation(inputfile)")

print(bitinformation)
