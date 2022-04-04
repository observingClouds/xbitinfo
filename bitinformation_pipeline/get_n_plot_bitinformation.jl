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

function main(filename::String)

    path = filename
    ncfile = NetCDF.open(path)

    nbits = 32

    coords = ("lat", "lon", "height_bnds", "height_2", "height", "time")
    v = ncfile.vars

    for c in coords
        delete!(v, c)
    end

    varnames = keys(v)  #["qv","temp","pres","qc","v","w","u","div","theta_v"]

    n = length(varnames)
    IC = fill(0.0,n,nbits)

    for (i,var) in enumerate(varnames)
        if length(size(ncfile.vars[var])) == 3
            X = ncfile.vars[var][:,1,:]
        elseif length(size(ncfile.vars[var])) == 2
            X = ncfile.vars[var][:,:]
        elseif length(size(ncfile.vars[var])) == 4
            X = ncfile.vars[var][:,:,:,1]
        end
        BitInformation.signed_exponent!(X)
        IC[i,:] = bitinformation(X,dim=1)
        print(var)
    end

    imshow(IC)
    xlabel("bits")                                                          # label x axis
    #xticks(0:31,vcat("+/-",[L"e_%$i" for i in 1:8],                           # label sign, exponent and mantissa bits
    #        [L"m_{%$i}" for i in 1:23]),fontsize=7);
    yticks(0:n-1, collect(varnames))
    tight_layout()
    savefig("bitinformation_surface_test2.pdf")

    # filter insignificant information via binomial distribution as explained in methods
    # for some variables the last bits do not have a free entropy of 1 bit, in that case
    # adjust the threshold by the information in the very last bits (which should always be 0)
    nvars = length(varnames)

    ICfilt = copy(IC)
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

    inflevel = 0.99
    infbits = [argmax(ICcsum_norm[i,:] .> inflevel) for i in 1:nvars]
    infbits100 = [argmax(ICcsum_norm[i,:] .> 0.999999999) for i in 1:nvars];

    # for plotting replace zeros with NaN to get white
    ICnan = copy(ICfilt)
    ICnan[iszero.(ICfilt)] .= NaN;


    # for plotting
    infbitsx = copy(vec(hcat(infbits,infbits)'))
    infbitsx100 = copy(vec(hcat(infbits100,infbits100)'))
    infbitsy = copy(vec(hcat(Array(0:nvars-1),Array(1:nvars))'));

    fig,ax1 = subplots(1,1,figsize=(8,5),sharey=true)
    ax1.invert_yaxis()
    tight_layout(rect=[0.06,0.18,0.93,0.98])
    pos = ax1.get_position()
    cax = fig.add_axes([pos.x0,0.12,pos.x1-pos.x0,0.02])

    ax1right = ax1.twinx()
    ax1right.invert_yaxis()

    # information
    cmap = ColorMap(ColorSchemes.turku.colors).reversed()
    pcm = ax1.pcolormesh(ICnan,vmin=0,vmax=1;cmap)
    cbar = colorbar(pcm,cax=cax,orientation="horizontal")
    cbar.set_label("information content [bit]")

    # 99% of real information enclosed
    ax1.plot(vcat(infbits,infbits[end]),Array(0:nvars),"C1",ds="steps-pre",zorder=10,label="99% of\ninformation")

    # grey shading
    ax1.fill_betweenx(infbitsy,infbitsx,fill(32,length(infbitsx)),alpha=0.4,color="grey")
    ax1.fill_betweenx(infbitsy,infbitsx100,fill(32,length(infbitsx)),alpha=0.1,color="c")
    ax1.fill_betweenx(infbitsy,infbitsx100,fill(32,length(infbitsx)),alpha=0.3,facecolor="none",edgecolor="c")

    # for legend only
    ax1.fill_betweenx([-1,-1],[-1,-1],[-1,-1],color="burlywood",label="last 1% of\ninformation",alpha=.5)
    ax1.fill_betweenx([-1,-1],[-1,-1],[-1,-1],facecolor="teal",edgecolor="c",label="false information",alpha=.3)
    ax1.fill_betweenx([-1,-1],[-1,-1],[-1,-1],color="w",label="unused bits")

    ax1.axvline(1,color="k",lw=1,zorder=3)
    ax1.axvline(9,color="k",lw=1,zorder=3)

    #grouplabls = ["Aerosols","Carbon oxides","Clouds & water",
    #                "Methane","Alkanes & alcohols I","Alkanes & alcohols II",
    #                "N & S oxides","Ozone","Others"]
    #
    #for (ig,group) in enumerate(groups)
    #    y = sum([length(g) for g in groups[1:ig]])
    #    ax1.axhline(y,color="w",lw=1.5,zorder=2)
    #    ax1.text(31.5,y-0.3,grouplabls[ig],color="w",ha="right",fontweight="bold")
    #end

    ax1.set_title("Real bitwise information content",loc="left",fontweight="bold")

    ax1.set_xlim(0,32)
    ax1.set_ylim(nvars,0)
    ax1right.set_ylim(nvars,0)

    ax1.set_yticks(Array(1:nvars).-0.5)
    ax1right.set_yticks(Array(1:nvars).-0.5)
    ax1.set_yticklabels(varnames)
    #ax1right.set_yticklabels([@sprintf "%4.1f" i for i in ICcsum[:,end]])
    ax1right.set_ylabel("total information per value [bit]")

    ax1.text(infbits[1]+0.1,0.8,"$(infbits[1]-9) mantissa bits",fontsize=8,color="saddlebrown")
    for i in 2:nvars
        ax1.text(infbits[i]+0.1,(i-1)+0.8,"$(infbits[i]-9)",fontsize=8,color="saddlebrown")
    end

    ax1.set_xticks([1,9])
    ax1.set_xticks(vcat(2:8,10:32),minor=true)
    ax1.set_xticklabels([])
    ax1.text(0,nvars+1.2,"sign",rotation=90)
    ax1.text(2,nvars+1.2,"exponent bits",color="darkslategrey")
    ax1.text(10,nvars+1.2,"mantissa bits")

    for i in 1:8
        ax1.text(i+.5,nvars+0.5,"$i",ha="center",fontsize=7,color="darkslategrey")
    end

    for i in 1:23
        ax1.text(8+i+.5,nvars+0.5,"$i",ha="center",fontsize=7)
    end

    # ax1.legend(loc=(0.725,0.27),framealpha=0.6) # 3D
    ax1.legend(loc=(0.725,0.62),framealpha=0.6) # surface
    savefig("bitinformation_surface_test3.pdf")

end


#bitinfo_dict = get_bitinformation("EUREC4A_DOM01_radiation_20200122T180000Z_latlon.nc")
#keepbits_dict = get_keepbits(bitinfo_dict, 0.99)
#print(json(keepbits_dict,4))
