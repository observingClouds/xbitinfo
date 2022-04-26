import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .bitinformation_pipeline import NMBITS, get_keepbits


def plot_bitinformation(bitinfo):
    """Plot bitwise information content.

    Inputs
    ------
    bitinfo : dict
      Dictionary containing the bitwise information content for each variable

    Returns
    -------
    fig : matplotlib figure

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> into_per_bit = bp.get_bitinformation(ds, dim="lon")
    >>> bp.plot_bitinformation(into_per_bit)
    <Figure size 1200x400 with 3 Axes>

    """
    import cmcrameri.cm as cmc

    nvars = len(bitinfo)
    varnames = bitinfo.keys()

    infbits_dict = get_keepbits(bitinfo, 0.99)
    infbits100_dict = get_keepbits(bitinfo, 0.999999999)

    ICnan = np.zeros((nvars, 64))
    infbits = np.zeros(nvars)
    infbits100 = np.zeros(nvars)
    ICnan[:, :] = np.nan
    for v, var in enumerate(varnames):
        ic = bitinfo[var]
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

    Inputs
    ------
    bitinfo : xr.Dataset
      raw input values for distributions
    nbints : int
      number of bins for histograms across all variable range. Defaults to 1000.
    cmap : str
      which matplotlib colormap to use. Defaults to "viridis".
    offset : float
      offset on the yaxis between variables 0 lines. Defaults to 0.01.
    close_zero : float
      threshold where to stop close to 0, when distributions ranges from negative to positive.
      Increase this value when seeing an unexpected dip around 0 in the distribution. Defaults to 0.01.

    Returns
    -------
    fig : matplotlib figure

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("eraint_uvz")
    >>> bp.plot_distribution(ds)
    <AxesSubplot:title={'center':'Statistical distributions'}, xlabel='value', ylabel='Probability density'>

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
