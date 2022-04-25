import matplotlib.pyplot as plt
import xarray as xr

from .bitinformation_pipeline import get_keepbits


def add_labels_fig3(
    ds,
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
    Helper function for visualization of in Klöwer et al. 2021.

    Klöwer, M., Razinger, M., Dominguez, J. J., Düben, P. D., & Palmer, T. N. (2021). Compressing atmospheric data into its real information content. Nature Computational Science, 1(11), 713–724. doi: 10/gnm4jj

    Inputs
    ------
    ds : xr.DataArray
      plotted before
    info_per_bit : dict
      Information content of each bit for each variable in ds. This is the output from get_bitinformation.
    inflevels : list of floats
      Level of information that shall be preserved.
    ax : plt.Axes or None
      axes. If None, get current axis.
    x_dim_name : str
      name of the x dimension. Defaults to "lon".
    y_dim_name : str
      name of the y dimension. Defaults to "lat".
    lon_coord_name : str
      name of the longitude coordinate. Only matters when plotting with multi-dimensional coordinates (i.e. curvilinear grids) with `cartopy` (when `transform=ccrs.Geodetic()` must be also set via `kwargs`). Defaults to x_dim_name.
    lat_coord_name : str
      name of the latitude coordinate. Only matters when plotting with multi-dimensional coordinates (i.e. curvilinear grids) with `cartopy` (when `transform=ccrs.Geodetic()` must be also set via `kwargs`). Defaults to y_dim_name.
    label_latitude :  float or str
      Latitude for the label. Defaults to "center", which uses the mean lat_coord_name.
    label_latitude_offset : float
      distance between `keepbits = int` and `x%` label. Defaults to 8.
    kwargs : dict
      kwargs to be passed to `ax.text` and `ax.plot`. Use `transform=ccrs.Geodetic()` when using `cartopy`

    Returns
    -------

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = bp.get_bitinformation(ds, dim="lon")
    >>> inflevels = [1.0, 0.9999, 0.99, 0.975, 0.95]
    >>> ds_bitrounded_along_lon = bp.bitround.bitround_along_dim(
    ...     ds, info_per_bit, dim="lon", inflevels=inflevels
    ... )
    >>> diff = (ds - ds_bitrounded_along_lon)["air"].isel(time=0)
    >>> diff.plot()  # doctest: +ELLIPSIS
    <matplotlib.collections.QuadMesh object at 0x7fc9ee61f730>
    >>> add_labels_fig3(diff, info_per_bit, inflevels)  # doctest: +ELLIPSIS

    Plotting an multi-dimensional coordinate dataset:
    >>> v = "Tair"
    >>> ds = xr.tutorial.load_dataset("rasm")
    >>> dim = "y"
    >>> bitinfo = bp.get_bitinformation(ds, dim=dim)
    >>> ds_bitrounded_along_lon = bp.bitround.bitround_along_dim(
    ...     ds, info_per_bit, dim=dim, inflevels=inflevels
    ... )
    >>> import cartopy.crs as ccrs  # doctest: +SKIP
    >>> fig, axis = plt.subplots(  # doctest: +SKIP
    ...     1, 1, subplot_kw=dict(projection=ccrs.PlateCarree())
    ... )
    >>> (ds - ds_bitrounded_along_lon)[v].isel(time=-10).plot(
    ...     ax=axis, transform=ccrs.PlateCarree()
    ... )  # doctest: +SKIP
    >>> add_lines_fig3(
    ...     (ds - ds_bitrounded_along_lon)[v].isel(time=0),
    ...     lon_coord_name="xc",
    ...     lat_coord_name="yc",
    ...     x_dim_name="x",
    ...     y_dim_name="y",
    ...     transform=ccrs.Geodetic(),
    ... )  # doctest: +SKIP

    """
    if lon_coord_name == "guess":
        lon_coord_name = x_dim_name
    if lat_coord_name == "guess":
        lat_coord_name = y_dim_name
    if label_latitude == "center":
        label_latitude = ds[lat_coord_name].mean()
    stride = ds[x_dim_name].size // len(inflevels)
    if ax is None:
        ax = plt.gca()

    for i, inf in enumerate(inflevels):
        # draw latitude line
        lons = ds.isel({x_dim_name: stride * i})[lon_coord_name]
        lats = ds.isel({x_dim_name: stride * i})[lat_coord_name]
        lons, lats = xr.broadcast(lons, lats)
        ax.plot(lons, lats, color="k", linewidth=1, **kwargs)
        # write inflevel
        t = ax.text(
            ds.isel(
                {
                    x_dim_name: int(stride * (i + 0.5)),
                    y_dim_name: ds[y_dim_name].size // 2,
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
            ds.isel(
                {
                    x_dim_name: int(stride * (i + 0.5)),
                    y_dim_name: ds[y_dim_name].size // 2,
                }
            )[lon_coord_name].values,
            label_latitude + label_latitude_offset,
            f"keepbits = {get_keepbits(info_per_bit, inf)[ds.name]}",
            horizontalalignment="center",
            color="k",
            **kwargs,
        )
        t_keepbits.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="white"))
