using BitInformation
using StatsBase

function get_bitinformation(X::AbstractArray{T}; kwargs...) where {T<:Base.IEEEFloat}

    BitInformation.signed_exponent!(X)
    IC = bitinformation(X; kwargs...)

    return IC

end


function get_bitinformation(X::AbstractArray{T}; kwargs...) where {T<:Union{Int16,Int32,Int64}}

    IC = bitinformation(X; kwargs...)

    return IC

end

function get_keepbits(bitinfo_dict)
    # filter insignificant information via binomial distribution as explained in methods
    # for some variables the last bits do not have a free entropy of 1 bit, in that case
    # adjust the threshold by the information in the very last bits (which should always be 0)
    bitinfo = bitinfo_dict["bitinfo"]
    maskinfo = bitinfo_dict["maskinfo"]
    inflevel = bitinfo_dict["inflevel"]

    ic = values(bitinfo)
    p = BitInformation.binom_confidence(maskinfo,0.99)  # get chance p for 1 (or 0) from binom distr
    M₀ = 1 - entropy([p,1-p],2)                            # free entropy of random 50/50 at trial size
    threshold = max(M₀,1.5*maximum(ic[end-3:end]))         # in case the information never drops to zero
                                                               # use something a bit bigger than maximum
                                                               # of the last 4 bits
    insigni = (ic .<= threshold) .& (collect(1:length(ic)) .> 9)
    ic[insigni] .= floatmin(Float64)

    # find bits with 99/100% of information
    ICcsum = cumsum(ic)
    ICcsum_norm = copy(ICcsum)
    ICcsum_norm[:] ./= ICcsum_norm[end]

    infbits = argmax(ICcsum_norm[:] .> inflevel)
    #infbits100 = [argmax(ICcsum_norm[i,:] .> 0.999999999) for i in 1:nvars];

    return infbits

end
