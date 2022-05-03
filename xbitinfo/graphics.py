import numpy as np
import xarray as xr

from .xbitinfo import NMBITS, get_keepbits


def add_bitinfo_labels(
    da,
    info_per_bit,
    inflevels,
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
    >>> ds_bitrounded_along_lon = xb.bitround.bitround_along_dim(
    ...     ds, info_per_bit, dim="lon", inflevels=inflevels
    ... )
    >>> diff = (ds - ds_bitrounded_along_lon)["air"].isel(time=0)
    >>> diff.plot()  # doctest: +ELLIPSIS
    <matplotlib.collections.QuadMesh object at ...>
    >>> add_bitinfo_labels(diff, info_per_bit, inflevels)  # doctest: +ELLIPSIS

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
    import matplotlib.pyplot as plt

    if lon_coord_name == "guess":
        lon_coord_name = x_dim_name
    if lat_coord_name == "guess":
        lat_coord_name = y_dim_name
    if label_latitude == "center":
        label_latitude = da[lat_coord_name].mean()
    stride = da[x_dim_name].size // len(inflevels)
    if ax is None:
        ax = plt.gca()

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

        # write keepbits
        t_keepbits = ax.text(
            da.isel(
                {
                    x_dim_name: int(stride * (i + 0.5)),
                    y_dim_name: da[y_dim_name].size // 2,
                }
            )[lon_coord_name].values,
            label_latitude + label_latitude_offset,
            f"keepbits = {get_keepbits(info_per_bit, inf)[da.name]}",
            horizontalalignment="center",
            color="k",
            **kwargs,
        )
        t_keepbits.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="white"))


def plot_bitinformation(bitinfo, cmap="turku"):
    """Plot bitwise information content as in Klöwer et al. 2021 Figure 2.

    Klöwer, M., Razinger, M., Dominguez, J. J., Düben, P. D., & Palmer, T. N. (2021).
    Compressing atmospheric data into its real information content.
    Nature Computational Science, 1(11), 713–724. doi: 10/gnm4jj

    Parameters
    ----------
    bitinfo : :py:func:`xarray.Dataset`
      Containing the bitwise information content for each variable
    cmap : str or plt.cm
      Colormap. Defaults to ``"turku"``.

    Returns
    -------
    fig : matplotlib figure

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = xb.get_bitinformation(ds, dim="lon")
    >>> xb.plot_bitinformation(info_per_bit)
    <Figure size 1200x400 with 3 Axes>

    """
    import matplotlib.pyplot as plt

    assert bitinfo.coords["dim"].shape <= (
        1,
    ), "Only bitinfo along one dimension is supported at the moment. Please select dimension before plotting."

    nvars = len(bitinfo)
    varnames = bitinfo.keys()

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
        infbits[v] = infbits_dict[var] + NMBITS[len(ic)]
        infbits100[v] = infbits100_dict[var] + NMBITS[len(ic)]
    ICnan = np.where(ICnan == 0, np.nan, ICnan)
    ICcsum = np.nancumsum(ICnan, axis=1)

    infbitsy = np.hstack([0, np.repeat(np.arange(1, nvars), 2), nvars])
    infbitsx = np.repeat(infbits, 2)
    infbitsx100 = np.repeat(infbits100, 2)

    fig_height = np.max([4, 4 + (nvars - 10) * 0.2])  # auto adjust to nvars
    fig, ax1 = plt.subplots(1, 1, figsize=(12, fig_height), sharey=True)
    ax1.invert_yaxis()
    ax1.set_box_aspect(1 / 32 * nvars)
    plt.tight_layout(rect=[0.06, 0.18, 0.8, 0.98])
    pos = ax1.get_position()
    cax = fig.add_axes([pos.x0, 0.12, pos.x1 - pos.x0, 0.02])

    ax1right = ax1.twinx()
    ax1right.invert_yaxis()
    ax1right.set_box_aspect(1 / 32 * nvars)

    if cmap == "turku":
        import cmcrameri.cm as cmc

        cmap = cmc.turku_r
    pcm = ax1.pcolormesh(ICnan, vmin=0, vmax=1, cmap=cmap)
    cbar = plt.colorbar(pcm, cax=cax, orientation="horizontal")
    cbar.set_label("information content [bit]")

    # 99% of real information enclosed
    ax1.plot(
        np.hstack([infbits, infbits[-1]]),
        np.arange(nvars + 1),
        "C1",
        ds="steps-pre",
        zorder=10,
        label="99% of\ninformation",
    )

    # grey shading
    ax1.fill_betweenx(
        infbitsy, infbitsx, np.ones(len(infbitsx)) * 32, alpha=0.4, color="grey"
    )
    ax1.fill_betweenx(
        infbitsy, infbitsx100, np.ones(len(infbitsx)) * 32, alpha=0.1, color="c"
    )
    ax1.fill_betweenx(
        infbitsy,
        infbitsx100,
        np.ones(len(infbitsx)) * 32,
        alpha=0.3,
        facecolor="none",
        edgecolor="c",
    )

    # for legend only
    ax1.fill_betweenx(
        [-1, -1],
        [-1, -1],
        [-1, -1],
        color="burlywood",
        label="last 1% of\ninformation",
        alpha=0.5,
    )
    ax1.fill_betweenx(
        [-1, -1],
        [-1, -1],
        [-1, -1],
        facecolor="teal",
        edgecolor="c",
        label="false information",
        alpha=0.3,
    )
    ax1.fill_betweenx([-1, -1], [-1, -1], [-1, -1], color="w", label="unused bits")

    ax1.axvline(1, color="k", lw=1, zorder=3)
    ax1.axvline(9, color="k", lw=1, zorder=3)

    fig.suptitle(
        "Real bitwise information content",
        x=0.05,
        y=0.98,
        fontweight="bold",
        horizontalalignment="left",
    )

    ax1.set_xlim(0, 32)
    ax1.set_ylim(nvars, 0)
    ax1right.set_ylim(nvars, 0)

    ax1.set_yticks(np.arange(nvars) + 0.5)
    ax1right.set_yticks(np.arange(nvars) + 0.5)
    ax1.set_yticklabels(varnames)
    ax1right.set_yticklabels([f"{i:4.1f}" for i in ICcsum[:, -1]])
    ax1right.set_ylabel("total information per value [bit]")

    ax1.text(
        infbits[0] + 0.1,
        0.8,
        f"{int(infbits[0]-9)} mantissa bits",
        fontsize=8,
        color="saddlebrown",
    )
    for i in range(1, nvars):
        ax1.text(
            infbits[i] + 0.1,
            (i) + 0.8,
            f"{int(infbits[i]-9)}",
            fontsize=8,
            color="saddlebrown",
        )

    ax1.set_xticks([1, 9])
    ax1.set_xticks(np.hstack([np.arange(1, 8), np.arange(9, 32)]), minor=True)
    ax1.set_xticklabels([])
    ax1.text(0, nvars + 1.2, "sign", rotation=90)
    ax1.text(2, nvars + 1.2, "exponent bits", color="darkslategrey")
    ax1.text(10, nvars + 1.2, "mantissa bits")

    for i in range(1, 9):
        ax1.text(
            i + 0.5, nvars + 0.5, i, ha="center", fontsize=7, color="darkslategrey"
        )

    for i in range(1, 24):
        ax1.text(8 + i + 0.5, nvars + 0.5, i, ha="center", fontsize=7)

    ax1.legend(bbox_to_anchor=(1.08, 0.5), loc="center left", framealpha=0.6)

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
    <AxesSubplot:title={'center':'Statistical distributions'}, xlabel='value', ylabel='Probability density'>

    """
    import matplotlib.pyplot as plt

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
    colors = plt.cm.get_cmap(cmap, nvars).colors

    for i in range(nvars):
        c = colors[i]
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
