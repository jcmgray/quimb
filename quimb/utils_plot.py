import functools
import math

# a style to use for matplotlib that works with light and dark backgrounds
NEUTRAL_STYLE = {
    "axes.edgecolor": (0.5, 0.5, 0.5),
    "axes.facecolor": (0, 0, 0, 0),
    "axes.grid": True,
    "axes.labelcolor": (0.5, 0.5, 0.5),
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.facecolor": (0, 0, 0, 0),
    "grid.alpha": 0.1,
    "grid.color": (0.5, 0.5, 0.5),
    "legend.frameon": False,
    "text.color": (0.5, 0.5, 0.5),
    "xtick.color": (0.5, 0.5, 0.5),
    "xtick.minor.visible": True,
    "ytick.color": (0.5, 0.5, 0.5),
    "ytick.minor.visible": True,
}


def default_to_neutral_style(fn):
    """Wrap a function or method to use the neutral style by default."""

    @functools.wraps(fn)
    def wrapper(
        *args,
        style="neutral",
        show_and_close=True,
        clear_previous=False,
        **kwargs
    ):
        import matplotlib.pyplot as plt

        if clear_previous:
            from IPython import display

            # clear old plots
            display.clear_output(wait=True)

        if style == "neutral":
            style = NEUTRAL_STYLE
        elif not style:
            style = {}

        with plt.style.context(style):
            out = fn(*args, **kwargs)

            if show_and_close:
                plt.show()
                plt.close()

            return out

    return wrapper


def _ensure_dict(k, v):
    import numpy as np
    from .schematic import hash_to_color, get_color

    # ensure is a dictionaty
    if not isinstance(v, dict):
        v = {"y": v}
    v["y"] = np.asarray(v["y"])

    if v["y"].size == 0:
        return None

    # make sure x-coords exists explicitly
    if "x" not in v:
        v["x"] = np.arange(v["y"].size)
    else:
        v["x"] = np.asarray(v["x"])

    # set label as data name by default
    v.setdefault("label", k)

    if v.get("color", None) is None:
        label = v["label"]
        if label is None:
            v["color"] = get_color("blue")
        else:
            v["color"] = hash_to_color(k, vmin=0.75, vmax=0.85)

    return v


@default_to_neutral_style
def plot_multi_series_zoom(
    data,
    zoom="auto",
    zoom_max=100,
    zoom_marker="|",
    zoom_markersize=3,
    xlabel="Iteration",
    figsize=None,
    **kwargs,
):
    """Plot possibly multiple series of data, using the asinh scale for an
    overview and a linear scale for a zoomed in final section.

    Parameters
    ----------
    data : dict[dict], dict[array], dict, array, optional
        The data to plot.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if isinstance(data, dict) and "y" not in data:
        # multiple plain, or configured, sequences supplied
        data = [_ensure_dict(k, v) for k, v in data.items()]
    else:
        # single plain, or configured, sequence supplied
        data = [_ensure_dict(None, data)]

    # remove any empty data
    data = [d for d in data if d is not None]

    nrows = len(data)

    if figsize is None:
        figsize = (8, 2 * nrows)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=figsize,
        width_ratios=(3, 2),
        gridspec_kw={"wspace": 0.05, "hspace": 0.10},
        squeeze=False,
    )

    n = max(d["x"][-1] for d in data)
    if zoom is not None:
        if zoom == "auto":
            zoom = min(zoom_max, n // 2)
    nz = n - zoom

    for i, d in enumerate(data):
        # get data and correct zoomed range
        x = d.pop("x")
        y = d.pop("y")
        iz = min(range(x.size), key=lambda i: x[i] < nz)

        label = d.pop("label")
        color = d.pop("color")
        yscale = d.pop("yscale", kwargs.get("yscale", "linear"))

        # plot overview
        ax = axs[i, 0]
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1,
        )
        if label is not None:
            ax.text(
                0.05,
                1.0,
                label,
                color=color,
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
        # x props
        ax.set_xscale("asinh", linear_width=20)
        ax.xaxis.set_major_locator(
            mpl.ticker.AsinhLocator(20, numticks=6, subs=range(10))
        )
        # y props
        ax.tick_params(axis="y", colors=color, which="both")
        if yscale == "linear":
            ax.yaxis.set_major_formatter(
                mpl.ticker.ScalarFormatter(useOffset=False)
            )
        else:
            ax.set_yscale(yscale)

        # highlight zoomed range
        ax.axvspan(nz, n, alpha=0.15, color=(0.5, 0.5, 0.5))

        # plot zoom
        ax = axs[i, 1]
        ax.plot(
            x[iz:],
            y[iz:],
            color=color,
            marker=zoom_marker,
            markersize=zoom_markersize,
        )
        # y props
        ax.yaxis.tick_right()
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.tick_params(axis="y", colors=color, which="both")
        if yscale == "linear":
            ax.yaxis.set_major_formatter(
                mpl.ticker.ScalarFormatter(useOffset=False)
            )
        else:
            ax.set_yscale(yscale)

    # remove ticklabels on all but last row
    for i in range(nrows - 1):
        axs[i, 0].tick_params(axis="x", labelbottom=False)
        axs[i, 1].tick_params(axis="x", labelbottom=False)

    # set x-limits to just cover full range of data
    for i in range(nrows):
        axs[i, 0].set_xlim(0.0 - 0.5, n + 0.5)
        axs[i, 1].set_xlim(nz - 0.5, n + 0.5)

    # make the xticklabels appear like [0, 1, 10, 100, ...]
    axs[-1, 0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axs[-1, 0].xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"
            if math.isclose(x, 0)
            or (x > 1 and math.isclose(math.log10(x) % 1, 0))
            else ""
        )
    )

    # set x-labels
    axs[-1, 0].set_xlabel(f"{xlabel} (full)")
    axs[-1, 1].set_xlabel(f"{xlabel} (zoom)")

    return fig, axs
