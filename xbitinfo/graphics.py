import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .xbitinfo import _cdf_from_info_per_bit, bit_partitioning, get_keepbits


def add_bitinfo_labels(
    da,
    info_per_bit,
    inflevels=None,
    keepbits=None,
    ax=None,
    x_dim_name="lon",
    y_dim_name="lat",
    lon_coord_name="guess",
    lat_coord_name="guess",
    label_latitude="center",
    label_latitude_offset=8,
    **kwargs,
):
    """
    Helper function for visualization of Figure 3 in Klöwer et al. 2021.
    Adds latitudinal lines and labels with keepbits and information content for each slice.

    Klöwer, M., Razinger, M., Dominguez, J. J., Düben, P. D., & Palmer, T. N. (2021).
    Compressing atmospheric data into its real information content. Nature Computational Science, 1(11), 713–724. doi: 10/gnm4jj

    Parameters
    ----------
    da : :py:func:`xarray.DataArray`
      Plotted data
    info_per_bit : dict
      Information content of each bit for each variable in ``da``. This is the output from :py:func:`xbitinfo.xbitinfo.get_bitinformation`.
    inflevels : list of floats
      Level of information that shall be preserved.
    ax : plt.Axes or None
      Axes. If ``None``, get current axis.
    x_dim_name : str
      Name of the x dimension. Defaults to ``"lon"``.
    y_dim_name : str
      Name of the y dimension. Defaults to ``"lat"``.
    lon_coord_name : str
      Name of the longitude coordinate. Only matters when plotting with multi-dimensional coordinates (i.e. curvilinear grids) with ``cartopy`` (when ``transform=ccrs.Geodetic()`` must be also set via ``kwargs``). Defaults to ``x_dim_name``.
    lat_coord_name : str
      Name of the latitude coordinate. Only matters when plotting with multi-dimensional coordinates (i.e. curvilinear grids) with ``cartopy`` (when ``transform=ccrs.Geodetic()`` must be also set via ``kwargs``). Defaults to ``y_dim_name``.
    label_latitude :  float or str
      Latitude for the label. Defaults to ``"center"``, which uses the mean ``lat_coord_name``.
    label_latitude_offset : float
      Distance between ``keepbits = int`` and ``x%`` label. Defaults to ``8``.
    kwargs : dict
      Kwargs to be passed to ``ax.text`` and ``ax.plot``. Use ``transform=ccrs.Geodetic()`` when using ``cartopy``

    Returns
    -------

    Example
    -------
    Plotting a single-dimension coordinate dataset:

    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> inflevels = [1.0, 0.9999, 0.99, 0.975, 0.95]
    >>> keepbits = None
    >>> ds_bitrounded_along_lon = xb.bitround.bitround_along_dim(
    ...     ds, info_per_bit, dim="lon", inflevels=inflevels
    ... )
    >>> diff = (ds - ds_bitrounded_along_lon)["air"].isel(time=0)
    >>> diff.plot()  # doctest: +ELLIPSIS
    <matplotlib.collections.QuadMesh object at ...>
    >>> add_bitinfo_labels(
    ...     diff, info_per_bit, inflevels, keepbits
    ... )  # doctest: +ELLIPSIS

    Plotting a multi-dimensional coordinate dataset

    >>> v = "Tair"
    >>> ds = xr.tutorial.load_dataset("rasm")
    >>> dim = "y"
    >>> info_per_bit = xb.get_bitinformation(ds, dim=dim)
    >>> ds_bitrounded_along_lon = xb.bitround.bitround_along_dim(
    ...     ds, info_per_bit, dim=dim, inflevels=inflevels
    ... )
    >>> import cartopy.crs as ccrs  # doctest: +SKIP
    >>> fig, axis = plt.subplots(  # doctest: +SKIP
    ...     1, 1, subplot_kw=dict(projection=ccrs.PlateCarree())
    ... )
    >>> (ds - ds_bitrounded_along_lon)[v].isel(time=-10).plot(
    ...     ax=axis, transform=ccrs.PlateCarree()
    ... )  # doctest: +SKIP
    >>> add_bitinfo_labels(
    ...     (ds - ds_bitrounded_along_lon)[v].isel(time=0),
    ...     lon_coord_name="xc",
    ...     lat_coord_name="yc",
    ...     x_dim_name="x",
    ...     y_dim_name="y",
    ...     transform=ccrs.Geodetic(),
    ... )  # doctest: +SKIP

    """

    if inflevels is None and keepbits is None:
        raise KeyError("Either inflevels or keepbits need to be provided")
    elif inflevels is not None and keepbits is not None:
        raise KeyError("Only inflevels or keepbits can be provided")
    if lon_coord_name == "guess":
        lon_coord_name = x_dim_name
    if lat_coord_name == "guess":
        lat_coord_name = y_dim_name
    if label_latitude == "center":
        label_latitude = da[lat_coord_name].mean()
    if ax is None:
        ax = plt.gca()

    dimension_dict = info_per_bit.dims
    dimension_list = list(dimension_dict.keys())
    dimension = dimension_list[0]
    CDF = _cdf_from_info_per_bit(info_per_bit, dimension)
    CDF_DataArray = CDF[da.name]

    data_type = np.dtype(dimension.replace("bit", ""))
    _, _, n_exp, _ = bit_partitioning(data_type)
    if inflevels is None:
        inflevels = []
        for i, keep in enumerate(keepbits):
            mantissa_index = keep + n_exp
            inflevels.append(CDF_DataArray[mantissa_index].values)

    if keepbits is None:
        keepbits = [get_keepbits(info_per_bit, ilev) for ilev in inflevels]

    if isinstance(keepbits, list) and all(
        isinstance(ds, xr.Dataset) for ds in keepbits
    ):
        keepbits_data = []
        for ds in keepbits:
            data_var = ds[da.name].values
            for value in data_var:
                keepbits_data.append(value)
        keepbits = keepbits_data

    stride = da[x_dim_name].size // len(inflevels)

    for i, inf in enumerate(inflevels):
        # draw latitude line
        lons = da.isel({x_dim_name: stride * i})[lon_coord_name]
        lats = da.isel({x_dim_name: stride * i})[lat_coord_name]
        lons, lats = xr.broadcast(lons, lats)
        ax.plot(lons, lats, color="k", linewidth=1, **kwargs)

        # write inflevel
        t = ax.text(
            da.isel(
                {
                    x_dim_name: int(stride * (i + 0.5)),
                    y_dim_name: da[y_dim_name].size // 2,
                }
            )[lon_coord_name].values,
            label_latitude - label_latitude_offset,
            str(round(inf * 100, 2)) + "%",
            horizontalalignment="center",
            color="k",
            **kwargs,
        )
        t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="white"))

    for i, keep in enumerate(keepbits):
        # write keepbits
        t_keepbits = ax.text(
            da.isel(
                {
                    x_dim_name: int(stride * (i + 0.5)),
                    y_dim_name: da[y_dim_name].size // 2,
                }
            )[lon_coord_name].values,
            label_latitude + label_latitude_offset,
            f"keepbits = {keep}",
            horizontalalignment="center",
            color="k",
            **kwargs,
        )
        t_keepbits.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="white"))


def split_dataset_by_dims(info_per_bit):
    """Split dataset by its dimensions.

    Parameters
    ----------
    info_per_bit : dict
      Information content of each bit for each variable in ``da``. This is the output from :py:func:`xbitinfo.xbitinfo.get_bitinformation`.

    Returns
    -------
    var_by_dim : dict
      Dictionary containing the dimensions of the datasets as keys and the dataset using the dimension as value.
    """
    var_by_dim = {d: [] for d in info_per_bit.dims}
    for var in info_per_bit.data_vars:
        assert (
            len(info_per_bit[var].dims) == 1
        ), f"Variable {var} has more than one dimension."
        var_by_dim[info_per_bit[var].dims[0]].append(var)
    return var_by_dim


def plot_bitinformation(bitinfo, information_filter=None, cmap="turku", crop=None):
    """Plot bitwise information content as in Klöwer et al. 2021 Figure 2.

    Klöwer, M., Razinger, M., Dominguez, J. J., Düben, P. D., & Palmer, T. N. (2021).
    Compressing atmospheric data into its real information content.
    Nature Computational Science, 1(11), 713–724. doi: 10/gnm4jj

    Parameters
    ----------
    bitinfo : :py:func:`xarray.Dataset`
      Containing the bitwise information content for each variable
    information_filter : str
      Filter algorithm to filter artificial information content. Defaults to ``None``.
      Available filters are: ``"Gradient"``.
    cmap : str or plt.cm
      Colormap. Defaults to ``"turku"``.
    crop : int
      Maximum bits to show in figure.

    Returns
    -------
    fig : matplotlib figure

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> xb.plot_bitinformation(info_per_bit)
    <Figure size 1200x800 with 4 Axes>

    """

    bitinfo = bitinfo.squeeze()
    assert (
        "dim" not in bitinfo.dims
    ), "Found dependence of bitinformation on dimension. Please reduce data first by e.g. `bitinfo.max(dim='dim')`"
    vars_by_dim = split_dataset_by_dims(bitinfo)
    bitinfo_all = bitinfo
    subfigure_data = [None] * len(vars_by_dim)
    for d, (dim, vars) in enumerate(vars_by_dim.items()):
        bitinfo = bitinfo_all[vars]
        data_type = np.dtype(dim.replace("bit", ""))
        n_bits, n_sign, n_exp, n_mant = bit_partitioning(data_type)
        nonmantissa_bits = n_bits - n_mant
        if crop is None:
            bits_to_show = n_bits
        else:
            bits_to_show = int(np.min([crop, n_bits]))
        nvars = len(bitinfo)
        varnames = list(bitinfo.keys())

        if information_filter == "Gradient":
            infbits_dict = get_keepbits(
                bitinfo,
                0.99,
                information_filter,
                **{"threshold": 0.7, "tolerance": 0.001},
            )
            infbits100_dict = get_keepbits(
                bitinfo,
                0.999999999,
                information_filter,
                **{"threshold": 0.7, "tolerance": 0.001},
            )
        else:
            infbits_dict = get_keepbits(bitinfo, 0.99)
            infbits100_dict = get_keepbits(bitinfo, 0.999999999)

        ICnan = np.zeros((nvars, 64))
        infbits = np.zeros(nvars)
        infbits100 = np.zeros(nvars)
        ICnan[:, :] = np.nan
        for v, var in enumerate(varnames):
            ic = bitinfo[var].squeeze(drop=True)
            ICnan[v, : len(ic)] = ic
            # infbits are all bits, infbits_dict were mantissa bits
            infbits[v] = infbits_dict[var] + nonmantissa_bits
            infbits100[v] = infbits100_dict[var] + nonmantissa_bits
        ICnan = np.where(ICnan == 0, np.nan, ICnan)
        ICcsum = np.nancumsum(ICnan, axis=1)

        infbitsy = np.hstack([0, np.repeat(np.arange(1, nvars), 2), nvars])
        infbitsx = np.repeat(infbits, 2)
        infbitsx100 = np.repeat(infbits100, 2)

        fig_height = np.max([4, 4 + (nvars - 10) * 0.2])  # auto adjust to nvars

        subfigure_data[d] = {}
        subfigure_data[d]["fig_height"] = fig_height
        subfigure_data[d]["nvars"] = nvars
        subfigure_data[d]["varnames"] = varnames
        subfigure_data[d]["ICnan"] = ICnan
        subfigure_data[d]["ICcsum"] = ICcsum
        subfigure_data[d]["infbits"] = infbits
        subfigure_data[d]["infbitsx"] = infbitsx
        subfigure_data[d]["infbitsy"] = infbitsy
        subfigure_data[d]["infbitsx100"] = infbitsx100
        subfigure_data[d]["nbits"] = (n_sign, n_exp, n_bits, n_mant, nonmantissa_bits)
        subfigure_data[d]["bits_to_show"] = bits_to_show

    fig_heights = [subfig["fig_height"] for subfig in subfigure_data]
    fig = plt.figure(figsize=(12, sum(fig_heights) + 2 * 2))
    fig_heights_incl_cax = fig_heights + [2 / (sum(fig_heights) + 2)] * 2
    grid = fig.add_gridspec(
        len(subfigure_data) + 2, 1, height_ratios=fig_heights_incl_cax
    )

    axs = []
    for i in range(len(subfigure_data) + 2):
        ax = fig.add_subplot(grid[i, 0])
        axs.append(ax)

    if isinstance(axs, plt.Axes):
        axs = [axs]

    fig.suptitle(
        "Real bitwise information content",
        x=0.05,
        y=0.98,
        fontweight="bold",
        horizontalalignment="left",
    )

    if cmap == "turku":
        import cmcrameri.cm as cmc

        cmap = cmc.turku_r

    max_bits_to_show = np.max([d["bits_to_show"] for d in subfigure_data])

    for d, subfig in enumerate(subfigure_data):
        infbits = subfig["infbits"]
        nvars = subfig["nvars"]
        n_sign, n_exp, n_bits, n_mant, nonmantissa_bits = subfig["nbits"]
        ICcsum = subfig["ICcsum"]
        ICnan = subfig["ICnan"]
        infbitsy = subfig["infbitsy"]
        infbitsx = subfig["infbitsx"]
        infbitsx100 = subfig["infbitsx100"]
        varnames = subfig["varnames"]
        bits_to_show = subfig["bits_to_show"]

        mbits_to_show = bits_to_show - nonmantissa_bits

        axs[d].invert_yaxis()
        axs[d].set_box_aspect(1 / max_bits_to_show * nvars)

        ax1right = axs[d].twinx()
        ax1right.invert_yaxis()
        ax1right.set_box_aspect(1 / max_bits_to_show * nvars)

        pcm = axs[d].pcolormesh(ICnan, vmin=0, vmax=1, cmap=cmap)

        if d == len(subfigure_data) - 1:
            cax = axs[len(subfigure_data)]
            lax = axs[len(subfigure_data) + 1]
            lax.axis("off")
            cbar = plt.colorbar(pcm, cax=cax, orientation="horizontal")
            cbar.set_label("information content [bit]")

        # 99% of real information enclosed
        l0 = axs[d].plot(
            np.hstack([infbits, infbits[-1]]),
            np.arange(nvars + 1),
            "C1",
            ds="steps-pre",
            zorder=10,
            label="99% of\ninformation",
        )

        # grey shading
        axs[d].fill_betweenx(
            infbitsy,
            infbitsx,
            np.ones(len(infbitsx)) * bits_to_show,
            alpha=0.4,
            color="grey",
        )
        axs[d].fill_betweenx(
            infbitsy,
            infbitsx100,
            np.ones(len(infbitsx)) * bits_to_show,
            alpha=0.1,
            color="c",
        )
        axs[d].fill_betweenx(
            infbitsy,
            infbitsx100,
            np.ones(len(infbitsx)) * bits_to_show,
            alpha=0.3,
            facecolor="none",
            edgecolor="c",
        )

        # for legend only
        l1 = axs[d].fill_betweenx(
            [-1, -1],
            [-1, -1],
            [-1, -1],
            color="burlywood",
            label="last 1% of\ninformation",
            alpha=0.5,
        )
        l2 = axs[d].fill_betweenx(
            [-1, -1],
            [-1, -1],
            [-1, -1],
            facecolor="teal",
            edgecolor="c",
            label="false information",
            alpha=0.3,
        )
        l3 = axs[d].fill_betweenx(
            [-1, -1], [-1, -1], [-1, -1], color="w", label="unused bits", edgecolor="k"
        )

        if n_sign > 0:
            axs[d].axvline(n_sign, color="k", lw=1, zorder=3)
        axs[d].axvline(nonmantissa_bits, color="k", lw=1, zorder=3)

        axs[d].set_ylim(nvars, 0)
        ax1right.set_ylim(nvars, 0)

        axs[d].set_yticks(np.arange(nvars) + 0.5)
        ax1right.set_yticks(np.arange(nvars) + 0.5)
        axs[d].set_yticklabels(varnames)
        ax1right.set_yticklabels([f"{i:4.1f}" for i in ICcsum[:, -1]])
        if d == len(subfigure_data) // 2:
            ax1right.set_ylabel("total information\nper value [bit]")

        axs[d].text(
            infbits[0] + 0.1,
            0.8,
            f"{int(infbits[0]-nonmantissa_bits)} mantissa bits",
            fontsize=8,
            color="saddlebrown",
        )
        for i in range(1, nvars):
            axs[d].text(
                infbits[i] + 0.1,
                (i) + 0.8,
                f"{int(infbits[i]-9)}",
                fontsize=8,
                color="saddlebrown",
            )

        major_xticks = np.array([n_sign, n_sign + n_exp, n_bits], dtype="int")
        axs[d].set_xticks(major_xticks[major_xticks <= bits_to_show])
        minor_xticks = np.hstack(
            [
                np.arange(n_sign, nonmantissa_bits - 1),
                np.arange(nonmantissa_bits, n_bits - 1),
            ]
        )
        axs[d].set_xticks(
            minor_xticks[minor_xticks <= bits_to_show],
            minor=True,
        )
        axs[d].set_xticklabels([])
        if n_sign > 0:
            axs[d].text(0, nvars + 1.2, "sign", rotation=90)
        if n_exp > 0:
            axs[d].text(
                n_sign + n_exp / 2,
                nvars + 1.2,
                "exponent bits",
                color="darkslategrey",
                horizontalalignment="center",
                verticalalignment="center",
            )
        axs[d].text(
            n_sign + n_exp + mbits_to_show / 2,
            nvars + 1.2,
            "mantissa bits",
            horizontalalignment="center",
            verticalalignment="center",
        )

        # Set xticklabels
        ## Set exponent labels
        for e, i in enumerate(range(n_sign, np.min([n_sign + n_exp, bits_to_show]))):
            axs[d].text(
                i + 0.5,
                nvars + 0.5,
                e + 1,
                ha="center",
                fontsize=7,
                color="darkslategrey",
            )
        ## Set mantissa labels
        for m, i in enumerate(
            range(n_sign + n_exp, np.min([n_sign + n_exp + n_mant, bits_to_show]))
        ):
            axs[d].text(i + 0.5, nvars + 0.5, m + 1, ha="center", fontsize=7)

        if d == len(subfigure_data) - 1:
            lax.legend(
                bbox_to_anchor=(0.5, 0),
                loc="center",
                framealpha=0.6,
                ncol=4,
                handles=[l0[0], l1, l2, l3],
            )
        axs[d].set_xlim(0, bits_to_show)

    fig.show()

    return fig


def plot_distribution(ds, nbins=1000, cmap="viridis", offset=0.01, close_zero=1e-2):
    """Plot statistical distributions of all variables as in Klöwer et al. 2021 Figure SI 1.
    For large data subsetting, i.e. ds = ds.isel(x=slice(None, None, 100)) is recommended.

    Klöwer, M., Razinger, M., Dominguez, J. J., Düben, P. D., & Palmer, T. N. (2021).
    Compressing atmospheric data into its real information content.
    Nature Computational Science, 1(11), 713–724. doi: 10/gnm4jj

    Parameters
    ----------
    bitinfo : :py:class:`xarray.Dataset`
      Raw input values for distributions
    nbints : int
      Number of bins for histograms across all variable range. Defaults to ``1000``.
    cmap : str
      Which matplotlib colormap to use. Defaults to ``"viridis"``.
    offset : float
      Offset on the yaxis between variables 0 lines. Defaults to ``0.01``.
    close_zero : float
      Threshold where to stop close to 0, when distributions ranges from negative to positive.
      Increase this value when seeing an unexpected dip around 0 in the distribution. Defaults to ``0.01``.

    Returns
    -------
    fig : matplotlib figure

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("eraint_uvz")
    >>> xb.plot_distribution(ds)
    <Axes: title={'center': 'Statistical distributions'}, xlabel='value', ylabel='Probability density'>

    """

    if not isinstance(ds, xr.Dataset):
        raise ValueError(
            f"plot_distribution(ds), requires xr.Dataset, found {type(ds)}"
        )

    varnames = list(ds.data_vars)
    nvars = len(varnames)
    ds = ds[varnames].squeeze()
    gmin, gmax = ds.to_array().min(), ds.to_array().max()
    f = 2  # factor for bounds
    if gmin < 0 and gmax > 0:
        bins_neg = np.geomspace(gmin * f, -close_zero, nbins // 2 + 1, dtype=float)
        bins_pos = np.geomspace(close_zero, gmax * f, nbins // 2, dtype=float)
        bins = np.concatenate([bins_neg, bins_pos], axis=-1)
    else:
        bins = np.geomspace(gmin / f, gmax * f, nbins + 1, dtype=float)

    H = np.zeros((nvars, nbins))
    for i, v in enumerate(varnames):
        d = ds[v].data.flatten()
        d = d[~np.isnan(d)]  # drop NaN
        H[i, :], _ = np.histogram(d, bins=bins, density=True)
        H[i, :] = H[i, :] / np.sum(H[i, :])  # normalize

    fig, ax = plt.subplots(1, 1, figsize=(5, 2 + nvars / 10))
    cmap_instance = plt.get_cmap(cmap, nvars)

    for i in range(nvars):
        c = cmap_instance.colors[i]
        plt.plot(bins[:-1], H[i, :] + offset * i, color=c)
        plt.fill_between(
            bins[:-1], H[i, :] + offset * i, offset * i, alpha=0.5, color=c
        )
    ax.set_xscale(
        "symlog"
    )  # https://stackoverflow.com/questions/43372499/plot-negative-values-on-a-log-scale
    ymax = max(0.05, nvars / 100 + 0.02)  # at least 10% y
    ax.set_ylim([-offset / 2, ymax])
    ax.set_xlim([bins[0], bins[-1]])
    minyticks = np.arange(0, ymax + 0.01, offset)
    majyticks = np.arange(0, ymax + 0.01, offset * 5)
    ax.set_yticks(minyticks, minor=True)
    ax.set_yticks(majyticks, minor=False)
    ax.set_yticklabels([str(int(i * 100)) + "%" for i in majyticks])

    axright = ax.twinx()
    axright.set_ylim([-offset / 2, ymax])
    axright.set_yticks(minyticks, minor=False)
    axright.set_yticklabels(
        varnames + [""] * (len(minyticks) - len(varnames)), minor=False
    )
    ax.set_xlabel("value")
    ax.set_ylabel("Probability density")
    ax.set_title("Statistical distributions")
    return ax
