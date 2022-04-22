import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from julia.api import Julia
from tqdm.auto import tqdm

jl = Julia(compiled_modules=False, debug=False)
from julia import Main  # noqa: E402

path_to_julia_functions = os.path.join(
    os.path.dirname(__file__), "get_n_plot_bitinformation.jl"
)
Main.path = path_to_julia_functions
jl.using("BitInformation")
jl.eval("include(Main.path)")


NMBITS = {64: 12, 32: 9, 16: 6}  # number of non mantissa bits for given dtype


def get_user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="filename of dataset (netCDF-file) whose information "
        "content should be retrieved",
        type=str,
    )
    args = parser.parse_args()
    return args


def get_bitinformation(ds, dim=None, axis=None, label=None, overwrite=False, **kwargs):
    """Wrap BitInformation.bitinformation().

    Inputs
    ------
    ds : xr.Dataset
      input netcdf to analyse
    dim : str
      Dimension over which to apply mean. Only one of the `dim` and `axis` arguments can be supplied.
    axis : int
      Axis over which to apply mean. Only one of the `dim` and `axis` arguments can be supplied.
    label : str
      label of the json to serialize bitinfo
    overwrite : bool
      if false, try using serialized bitinfo based on label; if true or label does not exist, run bitinformation
    kwargs
      to be passed to bitinformation:
      - masked_value: defaults to NaN (different to bitinformation.jl)
      - mask: use masked_value instead
      - set_zero_insignificant (bool): defaults to True
      - confidence (float): defaults to 0.99

    Returns
    -------
    info_per_bit : dict
      Information content per bit and variable

    Example
    -------
        >>> ds = xr.tutorial.load_dataset("air_temperature")
        >>> bp.get_bitinformation(ds, dim="lon")
        {'air': array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00, 3.94447851e-01, 3.94447851e-01, 3.94447851e-01,
               3.94447851e-01, 3.94447851e-01, 3.94310542e-01, 7.36739987e-01,
               5.62682836e-01, 3.60511555e-01, 1.52471111e-01, 4.18818055e-02,
               3.65276146e-03, 1.19975820e-05, 4.39366160e-05, 4.18329296e-05,
               2.54572089e-05, 1.44121797e-04, 1.34144798e-03, 1.55468479e-06,
               5.38601212e-04, 8.09862581e-04, 1.74893445e-04, 4.97915410e-05,
               3.88027711e-04, 0.00000000e+00, 3.95323228e-05, 6.88854435e-04])}
    """
    if overwrite:
        calc = True
    else:
        calc = False
        if label is None:
            calc = True
        else:
            info_per_bit = load_bitinformation(label)
            if info_per_bit is None:
                calc = True
    if calc:
        # check keywords
        if (axis is None and dim is None) or (axis is not None and dim is not None):
            raise ValueError(
                "Please provide either `axis` or `dim` but not both or none."
            )
        if axis:
            if not isinstance(axis, int):
                raise ValueError(f"Please provide `axis` as `int`, found {type(axis)}.")
        if dim:
            if not isinstance(dim, str):
                raise ValueError(f"Please provide `dim` as `str`, found {type(dim)}.")
        if "mask" in kwargs:
            raise ValueError(
                "`bitinformation_pipeline` does not wrap the mask argument. Mask your xr.Dataset with NaNs instead."
            )

        info_per_bit = {}
        for var in tqdm(ds.data_vars):
            X = ds[var].values
            Main.X = X
            if axis is not None:
                # in julia convention axis + 1
                dim = axis + 1
            if isinstance(dim, str):
                # in julia convention axis + 1
                dim = ds[var].get_axis_num(dim) + 1
            assert isinstance(dim, int)
            Main.dim = dim
            if "masked_value" not in kwargs:
                kwargs[
                    "masked_value"
                ] = f"convert({str(ds[var].dtype).capitalize()},NaN)"
            if "set_zero_insignificant" not in kwargs:
                kwargs["set_zero_insignificant"] = True
            kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            # convert python to julia bool
            kwargs_str = kwargs_str.replace("True", "true").replace("False", "false")
            logging.debug(f"get_bitinformation(X, dim={dim}, {kwargs_str})")
            info_per_bit[var] = jl.eval(
                f"get_bitinformation(X, dim={dim}, {kwargs_str})"
            )
        if label is not None:
            with open(label + ".json", "w") as f:
                logging.debug(f"Save bitinformation to {label+'.json'}")
                json.dump(info_per_bit, f, cls=JsonCustomEncoder)
    return info_per_bit


def load_bitinformation(label):
    """Load bitinformation from JSON file"""
    label_file = label + ".json"
    if os.path.exists(label_file):
        with open(label_file) as f:
            logging.debug(f"Load bitinformation from {label+'.json'}")
            info_per_bit = json.load(f)
        return info_per_bit
    else:
        return None


def get_keepbits(info_per_bit, inflevel=0.99):
    """Get the number of mantissa bits to keep. To be used in xr_bitround and jl_bitround.

    Inputs
    ------
    info_per_bit : dict
      Information content of each bit for each variable in ds. This is the output from get_bitinformation.
    inflevel : float or dict
      Level of information that shall be preserved. Of type `float` if the
      preserved information content should be equal across variables, otherwise of type `dict`.

    Returns
    -------
    keepbits : dict
      Number of mantissa bits to keep per variable

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = bp.get_bitinformation(ds, dim="lon")
    >>> bp.get_keepbits(info_per_bit)
    {'air': 7}
    >>> bp.get_keepbits(info_per_bit, inflevel=0.99999999)
    {'air': 14}
    """
    keepmantissabits = {}
    for v, ic in info_per_bit.items():
        # set below threshold to zero
        # use something a bit bigger than maximum of the last 4 bits
        threshold = 1.5 * np.max(ic[-4:])
        ic_over_threshold = np.where(ic < threshold, 0, ic)
        ic_over_threshold_cum = np.cumsum(ic_over_threshold)  # CDF
        # normed CDF
        ic_over_threshold_cum_normed = ic_over_threshold_cum / ic_over_threshold_cum[-1]
        # return mantissabits to keep therefore subtract sign and exponent bits
        il = inflevel[v] if isinstance(inflevel, dict) else inflevel
        keepmantissabits[v] = (
            np.argmax(ic_over_threshold_cum_normed > il) + 1 - NMBITS[len(ic)]
        )
    return keepmantissabits


def _get_keepbits(ds, info_per_bit, inflevel=0.99):
    """Get the amount of mantissa bits to keep for a given information content.

    Inputs
    ------
    ds : xr.Dataset
      Dataset for which the information content has been retrieved
    info_per_bit : dict
      Information content of each bit for each variable in ds. This is the output from get_bitinformation.
    inflevel : float or dict
      Level of information that shall be preserved. Of type `float` if the
      preserved information content should be equal across variables, otherwise of type `dict`.

    Returns
    -------
    keepbits : dict
      Number of mantissa bits to keep per variable

    Example
    -------
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> info_per_bit = bp.get_bitinformation(ds, dim="lon")
    >>> bp._get_keepbits(ds, info_per_bit)
    {'air': 7}
    >>> bp._get_keepbits(ds, info_per_bit, inflevel=0.99999999)
    {'air': 14}
    """

    def get_inflevel(var, inflevel):
        """Helper function to load inflevel depending on input type."""
        if isinstance(inflevel, dict):
            return inflevel[var]
        else:
            return inflevel

    keepbits = {}
    config = {}
    for var in ds.data_vars:
        config[var] = {
            "inflevel": get_inflevel(var, inflevel),
            "bitinfo": info_per_bit[var],
            "maskinfo": int(ds[var].notnull().sum()),
        }
        Main.config = config[var]
        keepbits[var] = jl.eval("get_keepbits(config)")
        # keep mantissa bits
        keepbits[var] = keepbits[var] - NMBITS[len(info_per_bit[var])]
    return keepbits


def _jl_bitround(X, keepbits):
    """Wrap BitInformation.round. Used in bp.jl_bitround."""
    Main.X = X
    Main.keepbits = keepbits
    return jl.eval("round!(X, keepbits)")


def plot_bitinformation(bitinfo):
    """Plot bitwise information content.

    Inputs
    ------
    bitinfo : dict
      Dictionary containing the bitwise information content for each variable

    Returns
    -------
    fig : matplotlib figure

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


def get_prefect_flow(paths=[]):
    """
    Create prefect.Flow for bitinformation_pipeline bitrounding paths.

    1. Analyse bitwise real information content
    2. Retrieve keepbits
    3. Apply bitrounding with `xr_bitround`
    4. Save as compressed netcdf with `to_compressed_netcdf`

    Many parameters can be changed when running the flow `flow.run(parameters=dict(chunk="auto"))`:
    - paths: list of Paths
        paths to be bitrounded
    - analyse_paths: str or int
        which paths to be passed to `bp.get_bitinformation`. choose from ["first_last", "all", int], where int is interpreted as stride, i.e. paths[::stride]. Defaults to "first".
    - label : see get_bitinformation
    - dim/axis : see get_bitinformation
    - inflevel : see get_keepbits
    - chunks : see https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html. Note that with `chunks=None`, `dask` is not used for I/O and the flow is still parallelized when using `DaskExecutor`.
    - overwrite : bool
        whether to overwrite bitrounded netcdf files. False (default) skips existing files.
    - complevel : see to_compressed_netcdf, defaults to 7.
    - rename : list
        replace mapping for paths towards new_path of bitrounded file, i.e. replace=[".nc", "_bitrounded_compressed.nc"]

    Inputs
    ------
    paths : list
      list of Paths of files to be processed by `get_bitinformation`, `get_keepbits`, `xr_bitround` and `to_compressed_netcdf`.

    Returns
    -------
    prefect.Flow
      see https://docs.prefect.io/core/concepts/flows.html#overview

    Example
    -------
    Imagine n files of identical structure, i.e. 1-year per file climate model output:
    >>> ds = xr.tutorial.load_dataset("rasm")
    >>> year, datasets = zip(*ds.groupby("time.year"))
    >>> paths = [f"{y}.nc" for y in year]
    >>> xr.save_mfdataset(datasets, paths)

    Create prefect.Flow and run sequentially
    >>> flow = bp.get_prefect_flow(paths=paths)
    >>> import prefect
    >>> logger = prefect.context.get("logger")
    >>> logger.setLevel("ERROR")
    >>> st = flow.run()

    Inspect flow state
    >>> # flow.visualize(st)  # requires graphviz

    Run in parallel with dask:
    >>> # import os  # https://docs.xarray.dev/en/stable/user-guide/dask.html
    >>> # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    >>> from prefect.executors import LocalDaskExecutor
    >>> executor = LocalDaskExecutor(scheduler="processes")
    >>> flow.run(executor=executor, parameters=dict(overwrite=True))
    <Success: "All reference tasks succeeded.">

    Modify parameters of a flow:
    >>> flow.run(parameters=dict(inflevel=0.9999), parameters=dict(overwrite=True))
    <Success: "All reference tasks succeeded.">

    See also
    --------
    - https://examples.dask.org/applications/prefect-etl.html
    - https://docs.prefect.io/core/getting_started/basic-core-flow.html

    """

    from prefect import Flow, Parameter, task, unmapped
    from prefect.engine.signals import SKIP

    from .bitround import xr_bitround

    @task
    def get_bitinformation_keepbits(
        paths,
        analyse_paths="first",
        label=None,
        inflevel=0.99,
        **get_bitinformation_kwargs,
    ):
        # take subset only for analysis in bitinformation
        if analyse_paths == "first_last":
            p = [paths[0], paths[-1]]
        elif analyse_paths == "all":
            p = paths
        elif analyse_paths == "first":
            p = paths[0]
        elif isinstance(analyse_paths, int):  # interpret as strides
            p = paths[::analyse_paths]
        else:
            raise ValueError(
                "Please provide analyse_paths as int or from ['first_last','all','first','last']."
            )
        ds = xr.open_mfdataset(p)
        info_per_bit = get_bitinformation(ds, label=label, **get_bitinformation_kwargs)
        keepbits = get_keepbits(info_per_bit, inflevel=inflevel)
        keepbits = {v: max(0, k) for v, k in keepbits.items()}  # ensure no negative
        return keepbits

    @task
    def bitround_and_save(
        path,
        keepbits,
        chunks=None,
        complevel=4,
        rename=[".nc", "_bitrounded_compressed.nc"],
        overwrite=False,
    ):
        new_path = path.replace(rename[0], rename[1])
        if not overwrite:
            if os.path.exists(new_path):
                try:
                    ds_new = xr.open_dataset(new_path, chunks=chunks)
                    ds = xr.open_dataset(path, chunks=chunks)
                    if (
                        ds.nbytes == ds_new.nbytes
                    ):  # bitrounded and original have same number of bytes in memory
                        raise SKIP(f"{new_path} already exists.")
                except Exception as e:
                    print(
                        f"{type(e)} when xr.open_dataset({new_path}), therefore delete and recalculate."
                    )
                    os.remove(new_path)
        ds = xr.open_dataset(
            path, chunks=chunks
        )  # .set_coords(vertices) # dont bitround vertices
        ds_bitround = xr_bitround(ds, keepbits)  # .reset_coords(vertices)
        ds_bitround.to_compressed_netcdf(new_path, complevel=complevel)
        return

    with Flow("bitinformation_pipeline") as flow:
        if paths == []:
            raise ValueError("Please provide paths of files to bitround, found [].")
        paths = Parameter("paths", default=paths)
        analyse_paths = Parameter("analyse_paths", default="first")
        dim = Parameter("dim", default=None)
        axis = Parameter("axis", default=0)
        inflevel = Parameter("inflevel", default=0.99)
        label = Parameter("label", default=None)
        rename = Parameter("rename", default=[".nc", "_bitrounded_compressed.nc"])
        overwrite = Parameter("overwrite", default=False)
        complevel = Parameter("complevel", default=7)
        chunks = Parameter("chunks", default=None)
        keepbits = get_bitinformation_keepbits(
            paths,
            analyse_paths=analyse_paths,
            dim=dim,
            axis=axis,
            inflevel=inflevel,
            label=label,
        )  # once
        bitround_and_save.map(
            paths,
            keepbits=unmapped(keepbits),
            rename=unmapped(rename),
            chunks=unmapped(chunks),
            complevel=unmapped(complevel),
            overwrite=unmapped(overwrite),
        )  # parallel map
    return flow


class JsonCustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (complex, np.complex)):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    args = get_user_input()
    ds = xr.open_mfdataset(args.filename)
    info_per_bit = get_bitinformation(ds, axis=0)
    print(info_per_bit)
    keepbits = get_keepbits(info_per_bit)
    print(keepbits)
