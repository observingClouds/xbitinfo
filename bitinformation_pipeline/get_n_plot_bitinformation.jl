using NetCDF, PyPlot, BitInformation, LaTeXStrings, JSON
using Statistics, StatsBase, ColorSchemes, Printf, PyPlot

function get_bitinformation(X::AbstractArray{T}, dim=1) where {T<:Base.IEEEFloat}

    BitInformation.signed_exponent!(X)
    IC = bitinformation(X,dim=dim)

    return IC

end


function get_bitinformation(X::AbstractArray{T}, dim=1) where {T<:Union{Int16,Int32,Int64}}

    IC = bitinformation(X,dim=dim)

    return IC

end


function get_keepbits(bitinfo_dict, inflevel=0.99)
    # filter insignificant information via binomial distribution as explained in methods
    # for some variables the last bits do not have a free entropy of 1 bit, in that case
    # adjust the threshold by the information in the very last bits (which should always be 0)
    nvars = length(bitinfo_dict)
    varnames = keys(bitinfo_dict)

    ICfilt = copy(reduce(vcat, transpose.(values(bitinfo_dict))))
    for i in 1:nvars
        ic = ICfilt[i,:]
        p = BitInformation.binom_confidence(900*440*137,0.99)  # get chance p for 1 (or 0) from binom distr
        M₀ = 1 - entropy([p,1-p],2)                            # free entropy of random 50/50 at trial size
        threshold = max(M₀,1.5*maximum(ic[end-3:end]))         # in case the information never drops to zero
                                                               # use something a bit bigger than maximum
                                                               # of the last 4 bits
        insigni = (ic .<= threshold) .& (collect(1:length(ic)) .> 9)
        ICfilt[i,insigni] .= floatmin(Float64)
    end


    # find bits with 99/100% of information
    ICcsum = cumsum(ICfilt,dims=2)
    ICcsum_norm = copy(ICcsum)
    for i in 1:nvars
        ICcsum_norm[i,:] ./= ICcsum_norm[i,end]
    end

    infbits_dict = Dict()
    for (i,var) in enumerate(varnames)
        infbits_dict[var] = argmax(ICcsum_norm[i,:] .> inflevel)
    end

    #infbits = [argmax(ICcsum_norm[i,:] .> inflevel) for i in 1:nvars]
    #infbits100 = [argmax(ICcsum_norm[i,:] .> 0.999999999) for i in 1:nvars];

    return infbits_dict

end
